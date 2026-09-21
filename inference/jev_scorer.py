"""Nimble typed-decision scorer: single forward pass, candidate-token logit readout.

Given a piece of state and a typed schema (choice / boolean / rubric score),
returns a calibrated probability for every option. No generation.

Device-adaptive: CUDA (bf16) or Apple Silicon MPS (bf16 weights, fp32 logits).
"""
import json
from pathlib import Path

import torch

MAX_CHOICES = 26


def _pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_model(model_id_or_path, device=None):
    """Load a merged (or base) model. Returns (model, tokenizer, device)."""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    dev, dtype = device or _pick_device()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id_or_path, dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(dev)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, dev


def score(model, tok, state, schema, temperature=1.0, max_input_tokens=4096):
    """Score one decision.

    state:  str (the context/evidence)
    schema: {"decision": {"description": ..., "type": "enum"|"boolean",
                          "choices": [...], "choice_descriptions": {...}}}
    temperature: divide logits by this before softmax (per-kind calibration)
    Returns {"prediction", "probabilities", "logits"} keyed by choice value.
    """
    from jev_schema import prepare_prompts, choice_key  # vendored prompt builder

    prepared = prepare_prompts(tok, state, schema, max_input_tokens)
    i = prepared.names.index("decision")
    ids = torch.tensor([prepared.full_ids[i]], device=next(model.parameters()).device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()[0]

    cand = prepared.candidate_ids[i]
    choices = prepared.choices[i]
    picked = logits[cand]
    scaled = picked / temperature
    probs = torch.softmax(scaled, -1)
    best = int(scaled.argmax())
    return {
        "prediction": choices[best],
        "probabilities": {choice_key(c): float(p) for c, p in zip(choices, probs.tolist())},
        "logits": {choice_key(c): float(v) for c, v in zip(choices, picked.tolist())},
    }


def apply_temperature(probabilities, logits, kind_temperature):
    """Re-calibrate an existing distribution: softmax(logits / T)."""
    import math
    mx = max(logits.values())
    exps = {k: math.exp(v / kind_temperature - mx / kind_temperature) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}