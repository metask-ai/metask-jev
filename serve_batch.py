"""metask-jev-4b production server — TypeSafe /v1/systemone compatible, dynamic batching.

POST /v1/systemone
  body: {"state": str|obj, "questions": {"decision": {type, instructions, criteria}}}
  resp: {"answers": {"decision": {"type", "probabilities"{...}, "noul": P(true)}}}
GET  /health -> {"ok": true, "qps_window", "p50_ms_window", "batch_filled"}

Design v2 (2026-09-22, after dual-replica experiment):
  Dual replicas on one GPU gave ZERO throughput gain (14.4 vs solo ~16 QPS
  theoretical): short-prompt prefill is GPU-compute-bound, so two replicas
  just interleave the same serial kernels. The real lever is batching:
  pad N waiting prompts to equal length, one forward for all N.

  Architecture: single model replica + collector thread.
    - HTTP threads enqueue (ids, meta) and wait on per-request events
    - collector drains the queue every 5 ms, pads to max len in the
      micro-batch (left padding), runs ONE forward, slices per-row logits
    - batch cap 16 (VRAM-safe at 4096 ctx: 16 x ~2k tokens ~ 6 GB activation)

  Expected: solo 63 ms/req -> batch-8 ~120 ms => 8/0.12 ≈ 65 QPS theoretical,
  30-40 QPS realistic with tokenization + HTTP overhead.

Usage (kunshan):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate nimble
  python serve_batch.py --port 8000 --model ~/nimble-small/runs/train/p4b-multilingual/merged
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
from jev_scorer import load_model  # noqa: E402
from jev_schema import prepare_prompts, choice_key  # noqa: E402

TEMPERATURE = {"choice": 1.9, "noul": 2.375, "score": 2.3}
BATCH_CAP = 16
COLLECT_WINDOW_S = 0.005

app = Flask(__name__)

_M = {"model": None, "tok": None, "dev": None}
_q: "queue.Queue" = queue.Queue()
_stats = {"n": 0, "window_start": time.time(), "batches": 0, "max_batch": 0,
          "lat": []}
_stats_lock = threading.Lock()


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
        _run_batch(batch)


def _run_batch(batch):
    tok, model, dev = _M["tok"], _M["model"], _M["dev"]
    t0 = time.perf_counter()
    # pad to the longest sequence in this micro-batch (left padding)
    maxlen = max(len(item["ids"]) for item in batch)
    pad_id = tok.pad_token_id
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["ids"])
        input_ids[i, maxlen - n:] = torch.tensor(item["ids"], dtype=torch.long)
        attn[i, maxlen - n:] = 1
    input_ids = input_ids.to(dev)
    attn = attn.to(dev)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn,
                    use_cache=False, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()

    for i, item in enumerate(batch):
        cand = item["candidate_ids"]
        picked = logits[i, cand]
        scaled = picked / item["temperature"]
        probs = torch.softmax(scaled, -1)
        best = int(scaled.argmax())
        item["result"] = {
            "prediction": item["choices"][best],
            "probabilities": {choice_key(c): float(p)
                              for c, p in zip(item["choices"], probs.tolist())},
        }
    dt = (time.perf_counter() - t0) * 1000
    with _stats_lock:
        _stats["batches"] += 1
        _stats["max_batch"] = max(_stats["max_batch"], len(batch))
        _stats["lat"].append(dt / len(batch))  # per-request amortized
        if len(_stats["lat"]) > 500:
            _stats["lat"] = _stats["lat"][-250:]
    for item in batch:
        item["event"].set()


def _score_async(state, schema, temperature):
    """Prepare ids on the calling thread (parallel across HTTP workers),
    enqueue for batched forward, wait for the result."""
    prepared = prepare_prompts(_M["tok"], state, schema, 4096)
    i = prepared.names.index("decision")
    item = {
        "ids": prepared.full_ids[i],
        "candidate_ids": prepared.candidate_ids[i],
        "choices": prepared.choices[i],
        "temperature": temperature,
        "event": threading.Event(),
        "result": None,
    }
    _q.put(item)
    if not item["event"].wait(30):
        raise TimeoutError("batched inference timeout")
    return item["result"]


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
        lat = sorted(_stats["lat"])
        p50 = lat[len(lat) // 2] * 1000 if lat else 0
        return jsonify(ok=True, qps_window=round(_stats["n"] / win, 2) if win else 0,
                       gpu_ms_per_req_p50=round(p50, 1),
                       batches=_stats["batches"], max_batch=_stats["max_batch"])


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
    args = ap.parse_args()

    print(f"[serve-batch] loading {args.model}", flush=True)
    _M["model"], _M["tok"], _M["dev"] = load_model(args.model)
    print(f"[serve-batch] ready on {_M['dev']}, batch cap {BATCH_CAP}", flush=True)
    threading.Thread(target=_collector, daemon=True, name="batch-collector").start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()