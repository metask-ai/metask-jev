#!/bin/bash
# metask-jev-4b 本机自测一键脚本 (Mac MPS / CUDA 自适应)
# 已装过环境: 直接复跑; 未装过: 先跑 install.sh
# 结果: ~/metask-jev/bench_results/ 逐条 jsonl + 末尾汇总表
set -uo pipefail

GREEN='\033[1;32m'; RED='\033[1;31m'; YEL='\033[1;33m'; NC='\033[0m'
say() { printf "\n${GREEN}[metask-jev]${NC} %s\n" "$*"; }
err() { printf "\n${RED}[metask-jev]${NC} %s\n" "$*"; }

DIR="$HOME/metask-jev"
RESULTS="$DIR/bench_results"
mkdir -p "$RESULTS"

# ---------- 环境 ----------
if [ ! -d "$DIR/.venv" ]; then
  say "未检测到安装 — 先执行一键安装:"
  echo "  curl -fsSL https://raw.githubusercontent.com/metask-ai/metask-jev/main/install.sh | bash"
  exit 1
fi
source "$DIR/.venv/bin/activate"
if ! python -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null; then
  err "transformers 缺 Qwen3_5 类 — 升级: pip install -U transformers"; exit 1
fi

# ---------- 设备 ----------
if command -v nvidia-smi >/dev/null 2>&1; then DEV="cuda"
elif [ "$(uname)" = "Darwin" ]; then DEV="mps"
else DEV="cpu"; fi
say "设备: $DEV"

# ---------- 任务文件 ----------
cd "$DIR"
for f in easy original hard; do
  if [ ! -s "$f.jsonl" ]; then
    say "下载 JevBench $f.jsonl..."
    curl -fsSL "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/$f.jsonl" -o "$f.jsonl"
  fi
done

# ---------- 跑分 (三层全量, 断点续跑) ----------
if [ ! -f "run_bench_local.py" ]; then
  say "下载评测 runner (run_bench_local.py)..."
  curl -fsSL "https://raw.githubusercontent.com/metask-ai/metask-jev/main/run_bench_local.py" -o "run_bench_local.py"
fi
say "开始跑分: easy 48 + judge 72 + hard 111 (断点续跑, MPS 约 25-40 分钟)"
python run_bench_local.py --tiers "easy judge hard" \
  --results-dir "$RESULTS" \
  --model wayfind/metask-jev-4b-policy-mix 2>&1 | grep -vE "Loading|warn|falling back"

# ---------- 汇总 ----------
echo
say "═══════════ 结果汇总 ═══════════"
printf "| %-8s | %-12s | %-9s |\n" "tier" "correct/att" "accuracy"
printf "|----------|--------------|----------|\n"
GC=0; GN=0
# tier 名 -> JevBench 官方结果文件名 (judge 层的文件叫 original.jsonl)
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
say "对照: kunshan CUDA 80.1% | Jev 1.13.0 75.3 | Nimble-9B 63.5"
say "逐条结果: $RESULTS/*.jsonl"