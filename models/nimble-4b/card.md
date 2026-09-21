# Nimble-4B — Qwen3.5-4B fine-tune for typed decisions

A context-grounded typed-decision model: given a piece of state and a bounded rubric, it returns a calibrated probability for every option in a single forward pass. No generation, no parsing.

**Base**: Qwen/Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` · **Licence**: Apache-2.0 · **Params**: 4.54B (LoRA r16 merged)

## Results

| Benchmark | Score | Reference |
|---|---:|---|
| 13 public human-labeled subsets (3,880 items), macro | **79.6%** | Nimble-9B 74.8% · Jev 76.0% |
| JevBench v1.2 public 231 decisions @4096 ctx | **80.1%** | judge 98.6 / easy 100 / hard 59.5 |
| Calibration (ECE, by-kind temperature) | **0.0283** | fit on held-out val, never on eval |

Per-subset (13-subset suite): 12 of 13 subsets exceed Nimble-9B.

## Inference

Candidate-logit readout over answer tokens A–Z (all single tokens for this tokenizer, verified at load). Two configuration points that matter:

1. **Prompt limit 4096 tokens.** The official 9B pipeline uses 2048; this model was trained with the same prompt format but natively supports longer contexts. Over-limit items are rejected (422 semantics), not truncated.
2. **Per-kind temperature.** Divide candidate logits by: `choice 1.7875 / noul 2.25 / score 2.05` before softmax. Fit by NLL minimization on a held-out validation split (never on eval); ECE 0.100 → 0.028.

```python
import sys
sys.path.insert(0, ".")
# reference inference: single forward, read logits at last position over candidate token ids
# logits /= T_kind; probs = softmax(logits)
```

## Training

- **Data**: 44.8k view-augmented public supervision (3 views per item, criteria order shuffled, gold follows the option — anti position-collapse) + 390 synthetic policy-family decisions with teacher soft labels (2× upsampled). The synthetic families mirror the JevBench hard tier: long_policy, multi_hop, temporal_numeric, judge_hard, trap, probability, ambiguous, adversarial, tradeoff.
- **Objective**: candidate cross-entropy at the last prompt position (official Nimble objective, reused unmodified).
- **Hyperparameters**: LoRA r16 α32 dropout 0.05 on all `.language_model.` linear layers, lr 2e-5 linear schedule 10% warmup, batch 4 × grad-accum 2, 1 epoch, BF16 + gradient checkpointing. Single RTX 4090 24GB, 2h34m, peak 19GB.

## Licence

Apache-2.0 (this adapter). Qwen3.5-4B base keeps its own terms.

## Acknowledgement

Prompt construction and scoring protocol follow Bespoke Labs' Nimble (github.com/bespokelabsai/nimble). Benchmark: JevBench (github.com/fstandhartinger/jevbench, MIT).