#!/bin/bash
# metask-jev-4b self-test: reproduce the published JevBench numbers on your machine.
# Device auto-detected (CUDA / Apple Silicon MPS / CPU). Resumable.
# Results: ~/metask-jev/bench_results/*.jsonl (per-item) + summary table below.
set -uo pipefail

GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
say() { printf "\n${GREEN}[metask-jev]${NC} %s\n" "$*"; }
err() { printf "\n${RED}[metask-jev]${NC} %s\n" "$*"; }

DIR="$HOME/metask-jev"
RESULTS="$DIR/bench_results"
mkdir -p "$RESULTS"

# ---------- environment ----------
if [ ! -d "$DIR/.venv" ]; then
  say "no install detected — run the one-command install first:"
  echo "  curl -fsSL https://raw.githubusercontent.com/metask-ai/metask-jev/main/install.sh | bash"
  exit 1
fi
source "$DIR/.venv/bin/activate"
if ! python -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null; then
  err "transformers is missing the Qwen3_5 class — upgrade: pip install -U transformers"
  exit 1
fi

# ---------- device ----------
if command -v nvidia-smi >/dev/null 2>&1; then DEV="cuda"
elif [ "$(uname)" = "Darwin" ]; then DEV="mps"
else DEV="cpu"; fi
say "device: $DEV"

# ---------- task files ----------
cd "$DIR"
for f in easy original hard; do
  if [ ! -s "$f.jsonl" ]; then
    say "fetching JevBench $f.jsonl..."
    curl -fsSL "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/$f.jsonl" -o "$f.jsonl"
  fi
done

# ---------- benchmark (3 tiers, resumable) ----------
if [ ! -f "run_bench_local.py" ]; then
  say "fetching the benchmark runner..."
  curl -fsSL "https://raw.githubusercontent.com/metask-ai/metask-jev/main/run_bench_local.py" -o "run_bench_local.py"
fi
say "running: easy 48 + judge 72 + hard 111 (resumable; ~25-40 min on MPS)"
python run_bench_local.py --tiers "easy judge hard" \
  --results-dir "$RESULTS" \
  --model wayfind/metask-jev-4b-policy-mix 2>&1 | grep -vE "Loading|warn|falling back"

# ---------- summary ----------
echo
say "═══════════ results ═══════════"
printf "| %-8s | %-12s | %-9s |\n" "tier" "correct/att" "accuracy"
printf "|----------|--------------|----------|\n"
GC=0; GN=0
# tier name -> JevBench task file name (the judge tier lives in original.jsonl)
tier_file() { case "$1" in easy) echo easy;; judge) echo original;; hard) echo hard;; esac; }

for pair in "easy:48" "judge:72" "hard:111"; do
  T=${pair%%:*}; N=${pair##*:}
  F=$(tier_file "$T")
  OK=0; C=0
  if [ -s "$RESULTS/$F.jsonl" ]; then
    while IFS= read -r line; do
      R=$(echo "$line" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['ok'], str(r['prediction'])==str(r['expected']))")
      [ "$(echo $R | cut -d' ' -f1)" = "True" ] && OK=$((OK+1)) && { [ "$(echo $R | cut -d' ' -f2)" = "True" ] && C=$((C+1)); }
    done < "$RESULTS/$F.jsonl"
  fi
  GN=$((GN+OK)); GC=$((GC+C))
  if [ "$OK" -gt 0 ]; then
    ACC=$(python3 -c "print(f'{$C/$OK*100:.1f}%')")
    printf "| %-8s | %-12s | %-9s |\n" "$T" "$C/$OK" "$ACC"
  else
    printf "| %-8s | %-12s | %-9s |\n" "$T" "skipped" "-"
  fi
done
printf "| %-8s | %-12s | %-9s |\n" "total" "$GC/$GN" "$(python3 -c "print(f'{$GC/$GN*100:.1f}%' if $GN else '-')")"
echo
say "published reference: 80.1% @4096 ctx (see the model card for the full tables)"
say "per-item results: $RESULTS/*.jsonl"