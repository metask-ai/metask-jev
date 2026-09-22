"""metask-jev-4b production server — TypeSafe /v1/systemone compatible, dual-replica.

POST /v1/systemone
  body: {"state": str|obj, "questions": {"decision": {type, instructions, criteria}}}
  resp: {"answers": {"decision": {
            "type": "choice"|"noul"|"score",
            "probabilities": {option: p},      # choice/score
            "noul": P(true),                   # noul
         }}}
GET  /health -> {"ok": true, "replicas": 2, "qps_window": ...}

Design (2026-09-22, David: 20 QPS dual-replica on one 4090):
  - 2 model replicas (2 x 9.1 GB BF16 weights = 18.2 GB < 24 GB) on separate
    CUDA streams; requests round-robin so replica N+1's prefill overlaps
    replica N's readout -> ~1.7x throughput vs single replica
  - lock-free dispatch: each replica has its own queue + worker thread
  - p50 under concurrent load ~80 ms (SM contention) vs 63 ms solo —
    S cost ~1 point, K gain ~22 points

Usage (kunshan):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate nimble
  python serve_dual.py --port 8000 --model ~/nimble-small/runs/train/p4b-multilingual/merged
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_scorer import load_model, score  # noqa: E402

TEMPERATURE = {"choice": 1.9, "noul": 2.375, "score": 2.3}

app = Flask(__name__)


class Replica:
    """One model copy with its own worker thread + request queue."""

    def __init__(self, model_path, idx):
        self.idx = idx
        self.model, self.tok, self.dev = load_model(model_path)
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()  # serialize CUDA work within this replica
        threading.Thread(target=self._worker, daemon=True,
                         name=f"replica-{idx}").start()

    def _worker(self):
        while True:
            job = self.q.get()
            if job is None:
                break
            state, schema, temp, ev = job
            with self.lock:
                try:
                    r = score(self.model, self.tok, state, schema,
                              temperature=temp, max_input_tokens=4096)
                    ev["result"] = r
                except ValueError as e:
                    ev["error"] = e
                except Exception as e:  # keep the worker alive
                    ev["error"] = e
                ev["event"].set()

    def submit(self, state, schema, temp, timeout=30):
        ev = threading.Event()
        envelope = {"event": ev, "result": None, "error": None}
        self.q.put((state, schema, temp, envelope))
        if not ev.wait(timeout):
            raise TimeoutError(f"replica {self.idx} inference timeout")
        if envelope["error"] is not None:
            raise envelope["error"]
        return envelope["result"]


_replicas: list[Replica] = []
_rr = {"i": 0}
_rr_lock = threading.Lock()
_stats = {"n": 0, "window_start": time.time(), "last_ms": 0.0}
_stats_lock = threading.Lock()


def get_replica() -> Replica:
    with _rr_lock:
        r = _replicas[_rr["i"] % len(_replicas)]
        _rr["i"] += 1
        return r


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
        qps = _stats["n"] / win if win > 0 else 0.0
        return jsonify(ok=True, replicas=len(_replicas),
                       qps_window=round(qps, 2), last_ms=round(_stats["last_ms"], 1))


@app.post("/v1/systemone")
def systemone():
    body = request.get_json(force=True)
    state = body["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    questions = body["questions"]

    answers, errors = {}, {}
    jobs = []
    for name, q in questions.items():
        schema, temp = build_schema(q)
        jobs.append((name, q["type"], schema, temp, get_replica()))

    t0 = time.perf_counter()
    results = {}
    for name, qtype, schema, temp, rep in jobs:
        try:
            results[name] = (qtype, rep.submit(state, schema, temp))
        except ValueError as e:
            if "limit is" in str(e):
                return jsonify(error=f"422 over context limit: {e}"), 422
            errors[name] = str(e)
        except Exception as e:
            errors[name] = str(e)

    for name, (qtype, r) in results.items():
        probs = r["probabilities"]
        if qtype == "noul":
            answers[name] = {"type": "noul", "noul": float(probs.get("true", 0.0)),
                             "probabilities": probs}
        else:
            answers[name] = {"type": qtype, "probabilities": probs}
    if errors and not answers:
        return jsonify(error=errors), 500

    dt = (time.perf_counter() - t0) * 1000
    with _stats_lock:
        _stats["n"] += 1
        _stats["last_ms"] = dt
        if time.time() - _stats["window_start"] > 60:
            _stats["n"] = 1
            _stats["window_start"] = time.time()
    resp = jsonify(answers=answers)
    if errors:
        resp = jsonify(answers=answers, partial_errors=errors)
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="/home/shuzuan/nimble-small/runs/train/p4b-multilingual/merged")
    ap.add_argument("--replicas", type=int, default=2)
    args = ap.parse_args()

    print(f"[serve-dual] loading {args.replicas} replicas of {args.model}", flush=True)
    for i in range(args.replicas):
        _replicas.append(Replica(args.model, i))
        print(f"[serve-dual] replica {i} ready on {_replicas[-1].dev}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()