"""Minimal demo: score one typed decision with a released nimble model."""
import argparse
from nimble_scorer import load_model, score

DEFAULTS = {
    # per-kind temperature (see models/<name>/temperature.json)
    "choice": 1.7875, "noul": 2.25, "score": 2.05,
}

state = ("The store accepts returns within 30 days of purchase. "
         "This item was bought 12 days ago and is unopened.")
schema = {"decision": {
    "description": "Is the item still eligible for return?",
    "type": "boolean",
    "choices": [False, True],
    "choice_descriptions": {"false": "Not eligible.", "true": "Eligible."},
}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Raymond1122/nimble-4b-policy-mix")
    args = ap.parse_args()
    model, tok, dev = load_model(args.model)
    r = score(model, tok, state, schema, temperature=DEFAULTS["noul"])
    print(f"device: {dev}")
    print(f"prediction: {r['prediction']}")
    for k, v in sorted(r["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")
