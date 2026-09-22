"""JevBench local runner (Mac MPS): score public tiers with metask-jev-4b.

Writes one jsonl per tier: {task_id, ok, correct, prediction, probabilities, latency_s}
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_scorer import load_model, score  # noqa: E402

TEMPERATURE = {"choice": 1.9, "noul": 2.375, "score": 2.3}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="easy judge hard")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model", default="wayfind/metask-jev-4b-policy-mix")
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    args = ap.parse_args()

    model, tok, dev = load_model(args.model)
    print(f"[bench] device={dev}", flush=True)

    base = Path(__file__).resolve().parent
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tier in args.tiers.replace(",", " ").split():
        fname = {"easy": "easy", "judge": "original", "hard": "hard"}.get(tier.strip(), tier.strip())
        src = base / f"{fname}.jsonl"
        if not src.exists():
            print(f"[skip] {fname}.jsonl not found", flush=True)
            continue
        rows = [json.loads(l) for l in src.open() if l.strip()]
        out = out_dir / f"{fname}.jsonl"
        done_ids = set()
        if out.exists():  # resume
            for line in out.open():
                if line.strip():
                    done_ids.add(json.loads(line)["task_id"])
        correct = ok = 0
        with out.open("a") as w:
            for i, row in enumerate(rows):
                if row["id"] in done_ids:
                    continue
                schema, T = build_schema(row["question"])
                state = row["state"] if isinstance(row["state"], str) \
                    else json.dumps(row["state"], ensure_ascii=False)
                t0 = time.perf_counter()
                try:
                    r = score(model, tok, state, schema, temperature=T,
                              max_input_tokens=args.max_input_tokens)
                    ok_flag, err = True, None
                except Exception as e:  # noqa: BLE001
                    ok_flag, err = False, f"{type(e).__name__}: {str(e)[:120]}"
                    r = {"prediction": None, "probabilities": {}}
                lat = time.perf_counter() - t0

                pred = r["prediction"]
                exp = row["expected"]
                if row["question"]["type"] == "noul":
                    if pred is True:
                        pred = "true"
                    elif pred is False:
                        pred = "false"
                    if exp == "yes":
                        exp = "true"
                    elif exp == "no":
                        exp = "false"
                correct_flag = (str(pred) == str(exp)) if ok_flag else False
                if ok_flag:
                    ok += 1
                    if correct_flag:
                        correct += 1
                w.write(json.dumps({
                    "task_id": row["id"], "ok": ok_flag, "correct": correct_flag,
                    "prediction": pred, "expected": exp,
                    "probabilities": r["probabilities"], "error": err,
                    "latency_s": round(lat, 4)}) + "\n")
                w.flush()
                if (i + 1) % 10 == 0:
                    acc = correct / ok * 100 if ok else 0
                    print(f"  [{fname}] {i+1}/{len(rows)}  acc {acc:.1f}%", flush=True)
        acc = correct / ok * 100 if ok else 0
        print(f"[done] {fname}: {correct}/{ok} = {acc:.1f}%", flush=True)


if __name__ == "__main__":
    main()