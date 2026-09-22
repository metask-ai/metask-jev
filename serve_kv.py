"""metask-jev-4b production server v4 — prefix-KV cache + dynamic batching.

POST /v1/systemone
  body: {"state": str|obj, "questions": {"decision": {...}, ...}}
  resp: {"answers": {name: {"type", "probabilities", "noul"}}}
GET  /health -> {"ok", "kv_entries", "kv_hits", "kv_tokens_cached", ...}

v4 (2026-09-22, David: 部署一套权重 + KV):
  The official prepare_prompts renders every field's prompt as
      start (system + state + schema JSON)  +  field-name  +  tail
  `start` is byte-identical across fields of the same request AND across
  requests that repeat the same state+schema. We cache `start`'s KV
  (HF DynamicCache) keyed by sha256(start-text), LRU capped by token budget:

    KV/token = 32 layers x 4 kv-heads x 256 head_dim x 2 (K,V) x 2 B = 128 KB
    10 GB budget = ~150 x 512-token states or ~19 x 4096-token states

  On a hit, prefill runs only the suffix (~10-40 tokens) against the cached
  KV — ~95% less prefill work. On a miss, full prefill, then the start-KV
  is stored. Multi-field requests reuse one prefill across fields (hit on
  fields 2..N by construction).

  Batching (v2) is kept for cache misses; cache hits run inline (tiny).
"""
import argparse
import copy
import hashlib
import json
import queue
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/shuzuan/nimble-small/third_party/nimble")
from jev_scorer import load_model  # noqa: E402
from jev_schema import prepare_prompts, choice_key  # noqa: E402

TEMPERATURE = {"choice": 1.9, "noul": 2.375, "score": 2.3}
BATCH_CAP = 16
COLLECT_WINDOW_S = 0.005
KV_BUDGET_TOKENS = 80_000   # 80k tokens x 128 KB = ~10 GB

app = Flask(__name__)
_M = {"model": None, "tok": None, "dev": None}
_q: "queue.Queue" = queue.Queue()
_kv: OrderedDict[str, tuple[DynamicCache, int]] = OrderedDict()  # key -> (cache, n_tokens)
_kv_lock = threading.Lock()
_stats = {"n": 0, "window_start": time.time(), "hits": 0, "misses": 0,
          "kv_tokens": 0, "lat": []}
_stats_lock = threading.Lock()


def _kv_put(key: str, cache: DynamicCache, n_tokens: int):
    with _kv_lock:
        _kv[key] = (cache, n_tokens)
        _stats["kv_tokens"] += n_tokens
        _kv.move_to_end(key)
        while _kv and _stats["kv_tokens"] > KV_BUDGET_TOKENS:
            _, (old, old_n) = _kv.popitem(last=False)
            _stats["kv_tokens"] -= old_n
            del old


def _kv_get(key: str):
    with _kv_lock:
        if key in _kv:
            _kv.move_to_end(key)
            cache, n = _kv[key]
            _stats["hits"] += 1
            return cache, n
        _stats["misses"] += 1
        return None


def _split_prompt(tok, state, schema, max_input_tokens):
    """Return (start_ids, {field: (full_ids, candidate_ids, choices)}).
    start_ids = token prefix shared by all fields (system+state+schema)."""
    prepared = prepare_prompts(tok, state, schema, max_input_tokens)
    names = prepared.names
    # recompute the shared start text exactly like jev_schema does
    import string as _string
    from jev_schema import choices_for, safe_json, SYSTEM_PROMPT
    fields = []
    for name in names:
        definition = schema[name]
        values = choices_for(definition)
        fields.append({
            "name": name, "description": definition["description"],
            "choices": [{"code": code, "value": value,
                         **({"description": definition["choice_descriptions"][choice_key(value)]}
                            if choice_key(value) in definition.get("choice_descriptions", {}) else {})}
                        for code, value in zip(_string.ascii_uppercase, values)],
        })
    marker = "__PARALLEL_FIELD_TARGET__"
    content = safe_json({"context": state, "schema": fields}) + "\n\nRequested field: " + marker
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    template = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    start_text, _ = template.rsplit(marker, 1)
    start_ids = tok.encode(start_text, add_special_tokens=False)
    # trim to the exact common token prefix across fields (BPE boundary safety)
    first = prepared.full_ids[0]
    n = 0
    while n < min(len(start_ids), len(first)) and start_ids[n] == first[n]:
        n += 1
    return start_ids[:n], names, prepared


def _prefill_to_cache(ids: list[int]):
    """One forward over ids; returns (DynamicCache, last_hidden)."""
    model, dev = _M["model"], _M["dev"]
    tokens = torch.tensor([ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=tokens, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].float()


def _decode_with_prefix(start_cache: DynamicCache, start_n: int, suffix_ids: list[int]):
    """Incremental forward: suffix tokens attend to cached prefix KV.
    Standard-attention K/V tensors are shared (read-only); linear-layer
    recurrent states are deep-copied so each request advances its own state
    (zero drift across hits)."""
    from transformers import DynamicCache as DC
    shared = DC()
    for li, layer in enumerate(start_cache.layers):
        if hasattr(layer, "keys") and layer.keys is not None:
            shared.update(layer.keys, layer.values, li)
        else:
            shared.layers.append(copy.deepcopy(layer))
    tokens = torch.tensor([suffix_ids], device=_M["dev"])
    with torch.no_grad():
        out = _M["model"](input_ids=tokens, past_key_values=shared, use_cache=False)
    return out.logits[:, -1, :].float()


def _run_batch(batch):
    """Cache-miss path: batched full prefill (v2 behaviour)."""
    tok, model, dev = _M["tok"], _M["model"], _M["dev"]
    t0 = time.perf_counter()
    maxlen = max(len(it["ids"]) for it in batch)
    pad_id = tok.pad_token_id or tok.eos_token_id
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, it in enumerate(batch):
        n = len(it["ids"])
        input_ids[i, maxlen - n:] = torch.tensor(it["ids"], dtype=torch.long)
        attn[i, maxlen - n:] = 1
    input_ids, attn = input_ids.to(dev), attn.to(dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn,
                    use_cache=False, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()
    dt = (time.perf_counter() - t0) * 1000
    with _stats_lock:
        _stats["lat"].append(dt / len(batch))
        if len(_stats["lat"]) > 500:
            _stats["lat"] = _stats["lat"][-250:]
    for i, it in enumerate(batch):
        p = torch.softmax(logits[i, it["candidate_ids"]] / it["temperature"], -1)
        best = int(p.argmax())
        it["result"] = {"prediction": it["choices"][best],
                        "probabilities": {choice_key(c): float(v)
                                          for c, v in zip(it["choices"], p.tolist())}}
        it["event"].set()


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


def _score_async(state, schema, temps):
    # schema values are already field definitions ({decision: {...}} wrapped);
    # flatten to what prepare_prompts expects: name -> definition
    flat = {name: definition["decision"] for name, definition in schema.items()}
    start_ids, names, prepared = _split_prompt(_M["tok"], state, flat, 4096)
    start_key = hashlib.sha256(
        json.dumps([state, json.dumps(schema, sort_keys=True)], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    results = {}
    for idx, name in enumerate(names):
        results[name] = _score_field_by_key(start_key, start_ids, prepared, idx, temps[name])
    return results


def _score_field_by_key(start_key: str, start_ids: list, prepared, idx: int, temperature: float):
    full_ids = prepared.full_ids[idx]
    cand = prepared.candidate_ids[idx]
    choices = prepared.choices[idx]
    hit = _kv_get(start_key)
    if hit is not None:
        cache, start_n = hit
        suffix = full_ids[start_n:]
        if suffix:
            logits = _decode_with_prefix(cache, start_n, suffix)
        else:
            logits = _prefill_to_cache(full_ids)[1]
    else:
        cache, logits = _prefill_to_cache(full_ids)
        # store shared-start KV + a deep copy of linear-layer recurrent states.
        # transformers 5.x DynamicCache: standard layers hold keys/values;
        # LinearAttention layers hold conv/recurrent state inside the layer obj.
        trimmed = DynamicCache()
        n_start = len(start_ids)
        for li, layer in enumerate(cache.layers):
            if hasattr(layer, "keys") and layer.keys is not None:
                trimmed.update(layer.keys[:, :, :n_start, :].clone(),
                               layer.values[:, :, :n_start, :].clone(), li)
            else:
                trimmed.layers.append(copy.deepcopy(layer))
        _kv_put(start_key, trimmed, n_start)
    p = torch.softmax(logits[0, cand] / temperature, -1)
    best = int(p.argmax())
    return {"prediction": choices[best],
            "probabilities": {choice_key(c): float(v)
                              for c, v in zip(choices, p.tolist())}}


def build_schema(q):
    qtype = q["type"]
    crit = q.get("criteria") or {}
    instr = q.get("instructions") or q.get("description") or "Decide."
    if qtype in ("noul", "boolean"):
        return {"decision": {
            "description": instr, "type": "boolean",
            "choices": [False, True],
            "choice_descriptions": {"false": crit.get("false", "No"),
                                    "true": crit.get("true", "Yes")}}}, TEMPERATURE["noul"]
    if qtype == "score":
        labels = [str(i) for i in range(len(crit))]
        return {"decision": {
            "description": instr, "type": "enum", "choices": labels,
            "choice_descriptions": dict(zip(labels, crit))}}, TEMPERATURE["score"]
    labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
    return {"decision": {
        "description": instr, "type": "enum", "choices": labels,
        "choice_descriptions": {k: (crit.get(k) or k) for k in labels}}}, TEMPERATURE["choice"]


@app.get("/health")
def health():
    with _stats_lock:
        win = time.time() - _stats["window_start"]
        lat = sorted(_stats["lat"])
        p50 = lat[len(lat) // 2] * 1000 if lat else 0
        return jsonify(ok=True, qps_window=round(_stats["n"] / win, 2) if win else 0,
                       kv_entries=len(_kv), kv_tokens=_stats["kv_tokens"],
                       kv_hits=_stats["hits"], kv_misses=_stats["misses"],
                       gpu_ms_per_req_p50=round(p50, 1))


@app.post("/v1/systemone")
def systemone():
    body = request.get_json(force=True)
    state = body["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    questions = body["questions"]
    schema = {}
    temps = {}
    for name, q in questions.items():
        schema[name], temps[name] = build_schema(q)  # returns {'decision': def}, T

    try:
        results = _score_async(state, schema, temps)
    except ValueError as e:
        if "limit is" in str(e):
            return jsonify(error=f"422 over context limit: {e}"), 422
        return jsonify(error=str(e)), 500
    except Exception as e:
        return jsonify(error=str(e)), 500

    answers = {}
    for name, q in questions.items():
        r = results[name]
        probs = r["probabilities"]
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": float(probs.get("true", 0.0)),
                             "probabilities": probs}
        else:
            answers[name] = {"type": q["type"], "probabilities": probs}
    with _stats_lock:
        _stats["n"] += 1
        if time.time() - _stats["window_start"] > 60:
            _stats["n"] = 1
            _stats["window_start"] = time.time()
    return jsonify(answers=answers)


def main():
    global KV_BUDGET_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="/home/shuzuan/nimble-small/runs/train/p4b-multilingual/merged")
    ap.add_argument("--kv-budget-tokens", type=int, default=KV_BUDGET_TOKENS)
    args = ap.parse_args()
    KV_BUDGET_TOKENS = args.kv_budget_tokens
    print(f"[serve-kv] loading {args.model}", flush=True)
    _M["model"], _M["tok"], _M["dev"] = load_model(args.model)
    print(f"[serve-kv] ready on {_M['dev']}, kv budget {KV_BUDGET_TOKENS} tokens "
          f"(~{KV_BUDGET_TOKENS*131072/1e9:.1f} GB)", flush=True)
    threading.Thread(target=_collector, daemon=True, name="batch-collector").start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
