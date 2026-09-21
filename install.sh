#!/bin/bash
# metask-jev one-command installer
# Creates a venv, installs pinned deps, downloads the model, verifies the
# A-Z single-token contract, and runs a self-test scoring example.
#
# Usage: curl -fsSL https://raw.githubusercontent.com/metask-ai/metask-jev/main/install.sh | bash
set -euo pipefail

MODEL_ID="wayfind/metask-jev-4b-policy-mix"
DIR="${HOME}/metask-jev"
REPO="https://github.com/metask-ai/metask-jev"

say() { printf "\n\033[1;31m[metask-jev]\033[0m %s\n" "$*"; }

# ---- 0. detect accelerator ----
GPU_HINT="CPU (slow but works)"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_HINT="NVIDIA GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
elif [ "$(uname)" = "Darwin" ]; then
  GPU_HINT="Apple Silicon (MPS)"
fi
say "accelerator: $GPU_HINT"

# ---- 1. python (needs >= 3.10: the Qwen3_5 model class lives in transformers 5.x) ----
PY=""
for cand in python3.14 python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    major=$("$cand" -c 'import sys; print(sys.version_info[0])')
    minor=$("$cand" -c 'import sys; print(sys.version_info[1])')
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then PY="$cand"; break; fi
  fi
done
if [ -z "$PY" ]; then
  say "ERROR: python >= 3.10 required (transformers 5.x needs it for the Qwen3_5 class)"
  exit 1
fi
say "python: $($PY --version)"

# ---- 2. venv ----
mkdir -p "$DIR" && cd "$DIR"
if [ ! -d ".venv" ]; then
  say "creating venv at $DIR/.venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. deps ----
say "installing torch + transformers (this may take a few minutes)"
if [ "$(uname)" = "Darwin" ]; then
  pip install -q --upgrade pip
  pip install -q "torch>=2.4" "transformers>=4.53" "accelerate" "huggingface_hub"
else
  pip install -q --upgrade pip
  pip install -q "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121 \
    || pip install -q "torch>=2.4"
  pip install -q "transformers>=4.53" "accelerate" "huggingface_hub"
fi

# ---- 4. inference code + benchmark runner ----
if [ ! -f "jev_scorer.py" ]; then
  say "fetching inference code from $REPO"
  for f in jev_scorer.py jev_schema.py; do
    curl -fsSL "$REPO/raw/main/inference/$f" -o "$f"
  done
fi
if [ ! -f "run_bench_local.py" ]; then
  say "fetching benchmark runner"
  curl -fsSL "$REPO/raw/main/run_bench_local.py" -o "run_bench_local.py"
fi

# ---- 5. model download (via huggingface_hub snapshot) ----
say "downloading $MODEL_ID (~8.5 GB, resumable)"
python - "$MODEL_ID" << 'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("model at:", p)
with open("model_path.txt", "w") as f:
    f.write(p)
EOF
MODEL_PATH=$(cat model_path.txt)

# ---- 6. verify A-Z single-token contract + self-test ----
say "verifying tokenizer contract and running self-test"
python - "$MODEL_PATH" << 'EOF'
import sys
from jev_scorer import load_model, score

model_path = sys.argv[1]
model, tok, dev = load_model(model_path)

# A-Z must each be a single token (candidate-logit contract)
bad = [chr(65+i) for i in range(26) if len(tok.encode(chr(65+i), add_special_tokens=False)) != 1]
assert not bad, f"tokenizer contract broken for: {bad}"
print(f"[ok] A-Z single-token contract verified on {dev}")

state = ("The store accepts returns within 30 days of purchase. "
         "This item was bought 12 days ago and is unopened.")
schema = {"decision": {
    "description": "Is the item still eligible for return?",
    "type": "boolean", "choices": [False, True],
    "choice_descriptions": {"false": "Not eligible.", "true": "Eligible."},
}}
r = score(model, tok, state, schema, temperature=2.25)  # noul temperature
print(f"[ok] self-test prediction: {r['prediction']}  probs: "
      + ", ".join(f"{k}={v:.3f}" for k, v in sorted(r['probabilities'].items(), key=lambda x: -x[1])))
EOF

say "install complete"
cat << TIP

  Quick use:
    source $DIR/.venv/bin/activate
    cd $DIR
    python - << 'PY'
from jev_scorer import load_model, score
model, tok, dev = load_model("$MODEL_PATH")
state = "Your state text here."
schema = {"decision": {"description": "Your question?",
    "type": "boolean", "choices": [False, True],
    "choice_descriptions": {"false": "No.", "true": "Yes."}}}
print(score(model, tok, state, schema, temperature=2.25))
PY

  Temperatures: choice 1.7875 / noul 2.25 / score 2.05 (see model card).
TIP