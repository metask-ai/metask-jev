"""metask-jev-4b production server — TypeSafe /v1/systemone compatible.

v3: length-bucketed CUDA Graph replay (port of H-worktree 26b, FEASIBILITY §3.9).

Architecture
  - requests are tokenized on the HTTP thread (HF tokenizers releases the GIL)
  - each request lands in a length bucket {256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}
  - per (bucket, batch-size) a CUDA graph is captured at startup:
      static buffers (input_ids / attention_mask / candidate_ids / candidate_mask),
      one replay = one whole forward for up to BATCH_CAP requests
  - collector drains the queue, groups by bucket, pads left into static buffers,
    replays, slices per-row candidate logits
  - bucket overflow (>4096 tokens) -> 422 per the official contract

Measured on kunshan 4090 (H-worktree data, 0.8B; 4B expected ~2.5x slower):
  eager bs1 28ms (CPU launch-bound, 3270 kernels) -> graph replay 12ms;
  bs4 graph 36.5ms/batch -> ~110 QPS @ bucket 512.

Usage (kunshan):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate nimble
  python serve_graph.py --port 8000 \
    --model ~/nimble-small/runs/train/p4b-multilingual/merged
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/shuzuan/nimble-small/third_party/nimble")
from jev_scorer import load_model  # noqa: E402
from jev_schema import prepare_prompts, choice_key  # noqa: E402
from nimble.training.schema_train import candidate_logits, MAX_CHOICES  # noqa: E402

TEMPERATURE = {"choice": 1.9, "noul": 2.375, "score": 2.3}
BUCKETS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
BATCH_CAP = 2          # 4B model: bs4 graphs OOM on 24GB (9.1GB weights + graph pools); bs2 fits
COLLECT_WINDOW_S = 0.004
WARM_BATCHES = 3

app = Flask(__name__)

_M = {"model": None, "tok": None, "dev": None}
_graphs: dict[tuple[int, int], object] = {}       # (bucket, bs) -> CUDAGraph
_static: dict[tuple[int, int], dict] = {}          # (bucket, bs) -> buffer dict
_q: "queue.Queue" = queue.Queue()
_stats = {"n": 0, "window_start": time.time(), "replays": 0, "max_batch": 0,
          "bucket_hist": {}}
_stats_lock = threading.Lock()
_overflow = {"flag": False, "msg": ""}


def _bucket_for(n_tokens: int) -> int:
    for b in BUCKETS:
        if n_tokens <= b:
            return b
    return 0  # overflow -> 422


def _capture_graph(bucket: int, bs: int):
    """Record one CUDA graph for (bucket, bs) with left-padded static buffers."""
    tok, model, dev = _M["tok"], _M["model"], _M["dev"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    buffers = {
        "input_ids": torch.full((bs, bucket), pad_id, dtype=torch.long, device=dev),
        "attention_mask": torch.zeros((bs, bucket), dtype=torch.long, device=dev),
        "candidate_ids": torch.zeros((bs, MAX_CHOICES), dtype=torch.long, device=dev),
        "candidate_mask": torch.zeros((bs, MAX_CHOICES), dtype=torch.bool, device=dev),
    }
    inputs = dict(buffers)

    def fill(rows):
        ids, mask, cands, cmask = (buffers["input_ids"], buffers["attention_mask"],
                                   buffers["candidate_ids"], buffers["candidate_mask"])
        ids.fill_(pad_id); mask.zero_(); cands.zero_(); cmask.zero_()
        for r, (tokens, candidates) in enumerate(rows):
            ids[r, bucket - len(tokens):] = torch.tensor(tokens, device=dev)
            mask[r, bucket - len(tokens):] = 1
            cands[r, :len(candidates)] = torch.tensor(candidates, device=dev)
            cmask[r, :len(candidates)] = True

    torch.cuda.empty_cache()  # free eager pool from previous capture
    # warmup on a side stream (required before capture)
    fill([(list(range(100, 100 + min(100, bucket))), list(range(32, 32 + 4)))] * bs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate_logits(model, inputs)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = candidate_logits(model, inputs)
    torch.cuda.empty_cache()  # release eager warmup pool; graphs keep private pools
    return graph, buffers, static_out, fill


def _collector():
    while True:
        first = _q.get()
        batch = [first]
        deadline = time.time() + COLLECT_WINDOW_S
        while len(batch) < BATCH_CAP and time.time() < deadline:
            try:
                batch.append(_q.get(timeout=max(0.0, deadline - time.time())))
            except queue.Empty:
                break

        # group by bucket; largest group first, remainder re-queued
        by_bucket: dict[int, list] = {}
        for item in batch:
            by_bucket.setdefault(item["bucket"], []).append(item)

        for bucket, items in by_bucket.items():
            while items:
                take = items[:BATCH_CAP]
                del items[:BATCH_CAP]
                _replay(bucket, take)


def _replay(bucket: int, items: list):
    model = _M["model"]
    bs = len(items)
    key = (bucket, bs)
    if key not in _graphs:
        # capture on demand (first time this shape is seen); pad group to a
        # captured bs if available to avoid shape explosion
        for cap_bs in sorted({BATCH_CAP, 2, 1}, reverse=True):
            if (bucket, cap_bs) in _graphs and bs < cap_bs:
                items += [items[-1]] * (cap_bs - bs)   # duplicate-pad, slice later
                bs = cap_bs
                key = (bucket, bs)
                break
        if key not in _graphs:
            try:
                _graphs[key], buffers, static_out, fill = _capture_graph(bucket, bs)
                _static[key] = {"buffers": buffers, "out": static_out, "fill": fill}
            except Exception as e:  # capture failed -> eager fallback for this shape
                _eager(bucket, items, error=e)
                return
    st = _static[key]
    rows = [(it["ids"], it["candidate_ids"]) for it in items]
    st["fill"](rows)
    st["out"]  # static tensors
    graph = _graphs[key]
    graph.replay()
    probs_all = st["out"].softmax(-1)
    for i, it in enumerate(items[:len(items)] if len(items) == bs else items):
        if i >= bs:
            break
        n = len(it["candidate_ids"])
        p = probs_all[i, :n]
        best = int(p.argmax())
        it["result"] = {
            "prediction": it["choices"][best],
            "probabilities": {choice_key(c): float(v)
                              for c, v in zip(it["choices"], p.tolist())},
        }
    for it in items:
        it["event"].set()
    with _stats_lock:
        _stats["replays"] += 1
        _stats["max_batch"] = max(_stats["max_batch"], bs)
        _stats["bucket_hist"][bucket] = _stats["bucket_hist"].get(bucket, 0) + 1


def _eager(bucket: int, items: list, error=None):
    """Fallback: per-request eager forward (also used before graph capture)."""
    model = _M["model"]
    dev = _M["dev"]
    for it in items:
        ids = torch.tensor([it["ids"]], device=dev)
        mask = torch.ones((1, len(it["ids"])), dtype=torch.long, device=dev)
        cands = torch.tensor([it["candidate_ids"] + [0] * (MAX_CHOICES - len(it["candidate_ids"]))],
                             device=dev)
        cmask = torch.tensor([[True] * len(it["candidate_ids"]) +
                              [False] * (MAX_CHOICES - len(it["candidate_ids"]))], device=dev)
        with torch.no_grad():
            out = candidate_logits(model, {"input_ids": ids, "attention_mask": mask,
                                           "candidate_ids": cands, "candidate_mask": cmask})
        p = out[0, :len(it["candidate_ids"])].softmax(-1)
        best = int(p.argmax())
        it["result"] = {
            "prediction": it["choices"][best],
            "probabilities": {choice_key(c): float(v)
                              for c, v in zip(it["choices"], p.tolist())},
        }
        it["event"].set()


def _score_async(state, schema, temperature):
    prepared = prepare_prompts(_M["tok"], state, schema, 4096)
    i = prepared.names.index("decision")
    n_tokens = len(prepared.full_ids[i])
    bucket = _bucket_for(n_tokens)
    item = {
        "ids": prepared.full_ids[i],
        "candidate_ids": prepared.candidate_ids[i],
        "choices": prepared.choices[i],
        "temperature": temperature,
        "bucket": bucket,
        "event": threading.Event(),
        "result": None,
    }
    if bucket == 0:
        raise ValueError(f"prompt is {n_tokens} tokens; limit is 4096")
    _q.put(item)
    if not item["event"].wait(60):
        raise TimeoutError("inference timeout")
    # per-kind temperature is applied here on the already-computed probs
    if temperature != 1.0:
        logits = {k: __import__("math").log(v) for k, v in item["result"]["probabilities"].items() if v > 0}
        mx = max(logits.values())
        exps = {k: __import__("math").exp(v / temperature - mx / temperature) for k, v in logits.items()}
        z = sum(exps.values())
        item["result"]["probabilities"] = {k: v / z for k, v in exps.items()}
        best = max(item["result"]["probabilities"], key=item["result"]["probabilities"].get)
        item["result"]["prediction"] = item["result"]["probabilities"][best] and _key_to_choice(item, best)
    return item["result"]


def _key_to_choice(item, key):
    # choice_key maps choice value -> "A"/"B"/...; invert via choices list order
    for idx, c in enumerate(item["choices"]):
        if choice_key(c) == key:
            return c
    return item["result"]["prediction"]


def build_schema(q):
    qtype = q["type"]
    crit = q.get("criteria") or {}
    if qtype == "noul":
        return {"decision": {
            "description": q["instructions"], "type": "boolean",
            "choices": [False, True],
            "choice_descriptions": {"false": crit.get("false", "No"),
                                    "true": crit.get("true", "Yes")}}}, TEMPERATURE["noul"]
    if qtype == "score":
        labels = [str(i) for i in range(len(crit))]
        return {"decision": {
            "description": q["instructions"], "type": "enum", "choices": labels,
            "choice_descriptions": dict(zip(labels, crit))}}, TEMPERATURE["score"]
    labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
    return {"decision": {
        "description": q["instructions"], "type": "enum", "choices": labels,
        "choice_descriptions": {k: (crit.get(k) or k) for k in labels}}}, TEMPERATURE["choice"]


@app.get("/health")
def health():
    with _stats_lock:
        win = time.time() - _stats["window_start"]
        return jsonify(ok=True, qps_window=round(_stats["n"] / win, 2) if win else 0,
                       graphs=len(_graphs), replays=_stats["replays"],
                       max_batch=_stats["max_batch"],
                       buckets={str(k): v for k, v in sorted(_stats["bucket_hist"].items())})


@app.post("/v1/systemone")
def systemone():
    body = request.get_json(force=True)
    state = body["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    questions = body["questions"]

    answers, errors = {}, {}
    for name, q in questions.items():
        schema, temp = build_schema(q)
        try:
            r = _score_async(state, schema, temp)
        except ValueError as e:
            if "limit is" in str(e):
                return jsonify(error=f"422 over context limit: {e}"), 422
            errors[name] = str(e)
            continue
        except Exception as e:
            errors[name] = str(e)
            continue
        probs = r["probabilities"]
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": float(probs.get("true", 0.0)),
                             "probabilities": probs}
        else:
            answers[name] = {"type": q["type"], "probabilities": probs}
    if errors and not answers:
        return jsonify(error=errors), 500
    resp = jsonify(answers=answers)
    if errors:
        resp = jsonify(answers=answers, partial_errors=errors)
    with _stats_lock:
        _stats["n"] += 1
        if time.time() - _stats["window_start"] > 60:
            _stats["n"] = 1
            _stats["window_start"] = time.time()
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="/home/shuzuan/nimble-small/runs/train/p4b-multilingual/merged")
    ap.add_argument("--warm-buckets", default="512,1024",
                    help="comma-separated buckets to pre-capture at startup")
    args = ap.parse_args()

    print(f"[serve-graph] loading {args.model}", flush=True)
    _M["model"], _M["tok"], _M["dev"] = load_model(args.model)
    print(f"[serve-graph] ready on {_M['dev']}", flush=True)

    # pre-capture the hot shapes so the first requests don't pay capture cost
    for b in [int(x) for x in args.warm_buckets.split(",")]:
        for bs in {BATCH_CAP, 1}:
            try:
                _graphs[(b, bs)], buffers, static_out, fill = _capture_graph(b, bs)
                _static[(b, bs)] = {"buffers": buffers, "out": static_out, "fill": fill}
                print(f"[serve-graph] captured (bucket={b}, bs={bs})", flush=True)
            except Exception as e:
                print(f"[serve-graph] capture failed (bucket={b}, bs={bs}): {e}", flush=True)

    threading.Thread(target=_collector, daemon=True, name="batch-collector").start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()