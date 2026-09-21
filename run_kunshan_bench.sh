#!/opt/homebrew/bin/bash
# metask-jev-4b 一键跑分（完整版）
# ==============================
# 流程: 环境自检 → 同步 adapter → 注册 cli → 逐层跑分(easy/judge/hard) → 汇总出分 → 同步回本地
#
# 用法:
#   bash run_kunshan_bench.sh              # 全量三层 (easy+judge+hard, ~30 分钟)
#   bash run_kunshan_bench.sh easy         # 只跑一层
#   bash run_kunshan_bench.sh all --fresh  # 清掉旧结果重跑
set -euo pipefail
cd "$(dirname "$0")"
source deploy/hosts.sh

ADDR=${HOSTS[kunshan]}
ROOT="/home/shuzuan/jevbench"  # jevbench harness 位置 (与 nimble-small 分离)
TIER="${1:-all}"
FRESH="${2:-}"
BENCH_USER=shuzuan
RESULTS_REMOTE="/home/$BENCH_USER/private"
RAW_REMOTE="$RESULTS_REMOTE/metask_bench_raw"

declare -A TIER_FILE=( [easy]=easy [judge]=original [hard]=hard )
declare -A TIER_N=( [easy]=48 [judge]=72 [hard]=111 )

if [ "$TIER" = "all" ]; then TIERS="easy judge hard"; else TIERS="$TIER"; fi

echo "╔══════════════════════════════════════════════╗"
echo "║  metask-jev-4b · JevBench 一键跑分           ║"
echo "║  host: kunshan (4090) · ctx 4096 · by_kind T ║"
echo "╚══════════════════════════════════════════════╝"

# ---------- [1/6] GPU 自检 ----------
echo
echo "── [1/6] kunshan GPU 空闲检查 ──"
GPU_USED=$(ssh "$ADDR" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
if [ "$GPU_USED" -lt 2000 ]; then
  echo "    OK: GPU free (${GPU_USED} MiB / 24564 MiB)"
else
  echo "    ✗ GPU busy (${GPU_USED} MiB) — kunshan 是生产机, 等待空闲后重试"
  exit 1
fi

# ---------- [2/6] 同步 PR adapter + 注册 ----------
echo "── [2/6] 同步 PR adapter + cli 注册 ──"
scp -q jevbench-fork/jevbench/adapters/metask_jev.py "$ADDR:$ROOT/jevbench/adapters/metask_jev.py"
ssh "$ADDR" "cd $ROOT/jevbench && python3 - << 'PYEOF'
from pathlib import Path
p = Path('jevbench/cli.py')
s = p.read_text()
changed = False
if 'MetaskJevAdapter' not in s:
    s = s.replace(
        'LayaLocalAdapter, Gliner2LocalAdapter, VerdictLocalAdapter, PawLocalAdapter,',
        'LayaLocalAdapter, Gliner2LocalAdapter, VerdictLocalAdapter, PawLocalAdapter, MetaskJevAdapter,')
    s = s.replace(
        '\"verdict_local\": VerdictLocalAdapter, \"paw_local\": PawLocalAdapter,',
        '\"verdict_local\": VerdictLocalAdapter, \"paw_local\": PawLocalAdapter, \"metask_jev\": MetaskJevAdapter,')
    s = s.replace(
        '\"laya_local\", \"gliner2_local\", \"verdict_local\", \"paw_local\", \"certo_local\"):',
        '\"laya_local\", \"gliner2_local\", \"verdict_local\", \"paw_local\", \"certo_local\", \"metask_jev\"):')
    s = s.replace(
        '\"verdict_local\", \"paw_local\", \"classifier_dev\", \"certo_local\"])',
        '\"verdict_local\", \"paw_local\", \"classifier_dev\", \"certo_local\", \"metask_jev\"])')
    p.write_text(s)
    changed = True
print('cli registered:', changed)
PYEOF"

# ---------- [3/6] 权重路径 + 旧结果清理 ----------
echo "── [3/6] 权重路径与结果目录 ──"
ssh "$ADDR" "export METASK_JEV_MODEL_PATH=/home/shuzuan/nimble-small/runs/train/p4b-policy-mix/merged; \
  test -f \$METASK_JEV_MODEL_PATH/model.safetensors && echo '    weights OK' || { echo '    ✗ weights missing'; exit 1; }; \
  mkdir -p $RAW_REMOTE $RESULTS_REMOTE"
if [ "$FRESH" = "--fresh" ]; then
  ssh "$ADDR" "rm -f $RESULTS_REMOTE/metask_bench_*.jsonl && echo '    old results cleared'"
fi

# ---------- [4/6] 逐层跑分 ----------
echo "── [4/6] 逐层跑分 ──"
for T in $TIERS; do
  F=${TIER_FILE[$T]}
  N=${TIER_N[$T]}
  echo "    ▶ $T ($N items)..."
  ssh "$ADDR" "cd $ROOT/jevbench && source ~/miniconda3/etc/profile.d/conda.sh && conda activate nimble && \
    export METASK_JEV_MODEL_PATH=/home/shuzuan/nimble-small/runs/train/p4b-policy-mix/merged && \
    python -m jevbench.cli run \
      --tasks datasets/public/$F.jsonl \
      --adapter metask_jev \
      --results $RESULTS_REMOTE/metask_bench_$T.jsonl \
      --raw-dir $RAW_REMOTE \
      --run-label metask-jev-4b \
      --cost-basis local_gpu_no_provider_tariff \
      2>&1 | grep -E '^\[jevbench\] (adapter|done)' || true"
done

# ---------- [5/6] 汇总出分 ----------
echo "── [5/6] 汇总 ──"
ssh "$ADDR" "python3 - << 'PYEOF'
import json, os
tiers = [('easy','easy',48), ('judge','original',72), ('hard','hard',111)]
rows = []
grand_c = grand_n = 0
print()
print('| tier | items | correct | accuracy |')
print('|---|---:|---:|---:|')
for label, fname, n_expected in tiers:
    path = f'/home/$BENCH_USER/private/metask_bench_{fname}.jsonl'
    if not os.path.exists(path):
        print(f'| {label} | {n_expected} | skipped | - |')
        continue
    ok = c = 0
    for line in open(path):
        r = json.loads(line)
        if r['ok']:
            ok += 1; c += r['correct']
    rows.append((label, ok, c))
    grand_c += c; grand_n += ok
    print(f'| {label} | {ok} | {c} | {c/ok*100:.1f}% |')
print(f'| **total** | {grand_n} | {grand_c} | **{grand_c/grand_n*100:.1f}%** |')
print()
lat = []
for label, fname, _ in tiers:
    path = f'/home/$BENCH_USER/private/metask_bench_{fname}.jsonl'
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r['ok'] and r.get('latency_s'):
                lat.append(r['latency_s'])
if lat:
    lat.sort()
    print(f'p50 latency: {lat[len(lat)//2]*1000:.0f} ms | n={len(lat)}')
PYEOF"

# ---------- [6/6] 收产物回本地 ----------
echo "── [6/6] 收产物回本地 ──"
mkdir -p internal/results_raw
for T in $TIERS; do
  F=${TIER_FILE[$T]}
  scp -q "$ADDR:$RESULTS_REMOTE/metask_bench_$F.jsonl" "internal/results_raw/" 2>/dev/null && echo "    metask_bench_$F.jsonl ✓"
done
echo
echo "╔══════════════════════════════════════════════╗"
echo "║  DONE — 结果在 internal/results_raw/         ║"
echo "╚══════════════════════════════════════════════╝"