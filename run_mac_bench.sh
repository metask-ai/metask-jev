#!/opt/homebrew/bin/bash
# metask-jev-4b 一键跑分 — Mac 本地 MPS 版
# ==========================================
# 在本机 Mac 上跑 JevBench 公开集 (easy/judge/hard)，MPS 推理。
# 前置: bash install.sh 已跑过 (~/metask-jev/.venv + 模型缓存)
#
# 用法:
#   bash run_mac_bench.sh              # 全量三层 (easy 48 + judge 72 + hard 111)
#   bash run_mac_bench.sh easy         # 只跑一层
#   bash run_mac_bench.sh all --fresh  # 清旧结果重跑
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="wayfind/metask-jev-4b-policy-mix"
RESULTS_DIR="$HOME/metask-jev/bench_results"
mkdir -p "$RESULTS_DIR"

declare -A TIER_FILE=( [easy]=easy [judge]=original [hard]=hard )
declare -A TIER_N=( [easy]=48 [judge]=72 [hard]=111 )
TIER="${1:-all}"
FRESH="${2:-}"
if [ "$TIER" = "all" ]; then TIERS="easy judge hard"; else TIERS="$TIER"; fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  metask-jev-4b · JevBench 本地跑分 (Mac MPS)     ║"
echo "╚══════════════════════════════════════════════════╝"

# ---- 环境检查 ----
if [ ! -d "$HOME/metask-jev/.venv" ]; then
  echo "✗ 未安装 — 先跑: bash install.sh"; exit 1
fi
source "$HOME/metask-jev/.venv/bin/activate"
python -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null \
  || { echo "✗ transformers 缺 Qwen3_5 类 — 重装: pip install -U transformers"; exit 1; }
echo "✓ 环境 OK (python $($PY --version 2>/dev/null || python --version))"

# ---- 任务文件 ----
for T in $TIERS; do
  F=${TIER_FILE[$T]}
  if [ ! -f "$F.jsonl" ]; then
    echo "── 下载 $F.jsonl ──"
    curl -fsSL "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/$F.jsonl" -o "$F.jsonl"
  fi
done

# ---- 逐层跑分 ----
if [ "$FRESH" = "--fresh" ]; then rm -f "$RESULTS_DIR"/*.jsonl; echo "✓ 旧结果已清"; fi

python run_bench_local.py --tiers "$TIERS" --results-dir "$RESULTS_DIR" --model "$MODEL_ID"

# ---- 汇总 ----
echo
echo "════════ 汇总 ════════"
python - << 'EOF'
import json, glob, os
tiers = [("easy","easy",48), ("judge","original",72), ("hard","hard",111)]
grand_c = grand_n = 0
print("| tier | items | correct | accuracy |")
print("|---|---:|---:|---:|")
lat = []
for label, fname, n_exp in tiers:
    path = os.path.expanduser(f"~/metask-jev/bench_results/{fname}.jsonl")
    if not os.path.exists(path):
        print(f"| {label} | {n_exp} | skipped | - |"); continue
    ok = c = 0
    for line in open(path):
        r = json.loads(line)
        if r.get("ok"):
            ok += 1
            if r.get("correct"): c += 1
    grand_c += c; grand_n += ok
    print(f"| {label} | {ok} | {c} | {c/ok*100:.1f}% |")
if grand_n:
    print(f"| **total** | {grand_n} | {grand_c} | **{grand_c/grand_n*100:.1f}%** |")
print(f"\n参考: kunshan CUDA @4096 = 80.1% | Jev 1.13.0 = 75.3")
EOF