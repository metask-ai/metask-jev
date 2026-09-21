"""metask-jev-4b HTTP server — TypeSafe /v1/systemone compatible.

POST /v1/systemone
  body: {"state": str|obj, "questions": {"decision": {type, instructions, criteria}}}
  resp: {"answers": {"decision": {
            "type": "choice"|"noul"|"score",
            "probabilities": {option: p},      # choice/score
            "noul": P(true),                   # noul
         }}}
GET  /health -> {"ok": true}
"""
import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_scorer import load_model, score  # noqa: E402

TEMPERATURE = {"choice": 1.7875, "noul": 2.25, "score": 2.05}

app = Flask(__name__)
_state = {}


def get_model(model_path):
    if "model" not in _state:
        _state["model"], _state["tok"], _state["dev"] = load_model(model_path)
        print(f"[serve] model loaded on {_state['dev']}", flush=True)
    return _state["model"], _state["tok"]


@app.route("/health")
def health():
    return jsonify(ok=True, model="metask-jev-4b")


@app.route("/v1/systemone", methods=["POST"])
def systemone():
    body = request.get_json(force=True)
    model, tok = get_model(_state.get("model_path", ""))
    questions = body.get("questions", {})
    state = body.get("state", "")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)

    answers = {}
    for name, q in questions.items():
        qtype = q["type"]
        crit = q.get("criteria") or {}
        if qtype == "noul":
            schema = {"decision": {
                "description": q.get("instructions", ""),
                "type": "boolean", "choices": [False, True],
                "choice_descriptions": {"false": crit.get("false", "No"),
                                        "true": crit.get("true", "Yes")}}}
            T = TEMPERATURE["noul"]
        elif qtype == "score":
            labels = [str(i) for i in range(len(crit))]
            schema = {"decision": {
                "description": q.get("instructions", ""),
                "type": "enum", "choices": labels,
                "choice_descriptions": dict(zip(labels, crit))}}
            T = TEMPERATURE["score"]
        else:
            labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            schema = {"decision": {
                "description": q.get("instructions", ""),
                "type": "enum", "choices": labels,
                "choice_descriptions": {k: (crit.get(k) or k) for k in labels}}}
            T = TEMPERATURE["choice"]

        try:
            r = score(model, tok, state, schema, temperature=T, max_input_tokens=4096)
        except ValueError as e:
            if "limit is" in str(e):
                return jsonify(error=f"422 over context limit: {e}"), 422
            return jsonify(error=str(e)), 500
        probs = r["probabilities"]
        if qtype == "noul":
            answers[name] = {"type": "noul", "noul": float(probs.get("true", 0.0)),
                             "probabilities": probs}
        else:
            answers[name] = {"type": qtype, "probabilities": probs}
    return jsonify(answers=answers)


def build_prompt(tok, state, schema, max_input_tokens):
    """Re-exported for card Quickstart: vendored official prepare_prompts."""
    from jev_schema import prepare_prompts
    return prepare_prompts(tok, state, schema, max_input_tokens)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="wayfind/metask-jev-4b-policy-mix",
                    help="HF repo id or path to model_path.txt")
    args = ap.parse_args()
    mp = Path(args.model)
    if mp.exists() and mp.is_file():  # model_path.txt written by install.sh
        args.model = mp.read_text().strip()
    _state["model_path"] = args.model
    print(f"[serve] starting on :{args.port}, model={args.model}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)