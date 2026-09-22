"""metask-jev-4b MPS front-end — round-robin proxy over N worker servers.

Workers run serve_graph.py on distinct ports under the MPS daemon
(nvidia-cuda-mps-control), so their kernels genuinely overlap on one GPU
instead of time-slicing. This front-end (no nginx dependency) listens on
the public port, round-robins POSTs, and merges /health.

Usage (kunshan):
  # 1. start MPS daemon once:
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
  nvidia-cuda-mps-control -d
  # 2. start workers (each loads its own replica under MPS):
  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps python serve_graph.py --port 8100 --replicas-skip-warm ...
  # 3. start this proxy:
  python serve_mps_front.py --port 8000 --workers 8100,8101
"""
import argparse
import itertools
import json
import threading
import time
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

app = Flask(__name__)
_workers: list[str] = []
_rr = itertools.count()
_rr_lock = threading.Lock()
_stats = {"n": 0, "window_start": time.time(), "err": 0}
_stats_lock = threading.Lock()


def _pick() -> str:
    with _rr_lock:
        return _workers[next(_rr) % len(_workers)]


def _proxy(path: str, body: bytes | None = None, method: str = "GET"):
    url = _pick() + path
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        with _stats_lock:
            _stats["err"] += 1
        return 502, json.dumps({"error": f"worker unreachable: {e}"}).encode()


@app.get("/health")
def health():
    merged = {"ok": True, "workers": []}
    for w in _workers:
        try:
            with urllib.request.urlopen(w + "/health", timeout=5) as r:
                merged["workers"].append(json.loads(r.read()))
        except Exception as e:
            merged["workers"].append({"error": str(e)})
    with _stats_lock:
        win = time.time() - _stats["window_start"]
        merged["front_qps"] = round(_stats["n"] / win, 2) if win else 0
    return jsonify(merged)


@app.post("/v1/systemone")
def systemone():
    body = request.get_data()
    code, data = _proxy("/v1/systemone", body=body, method="POST")
    with _stats_lock:
        _stats["n"] += 1
        if time.time() - _stats["window_start"] > 60:
            _stats["n"] = 1
            _stats["window_start"] = time.time()
    return app.response_class(data, status=code, mimetype="application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", default="8100,8101")
    args = ap.parse_args()
    _workers.extend(f"http://127.0.0.1:{p.strip()}" for p in args.workers.split(","))
    print(f"[mps-front] proxying :{args.port} -> {_workers}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()