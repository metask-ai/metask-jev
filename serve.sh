#!/bin/bash
# serve.sh — metask-jev-4b 一键启动 TypeSafe 兼容 HTTP 服务
# POST /v1/systemone  {state, questions:{decision:{type,instructions,criteria}}}
#   -> {"answers":{"decision":{"type":..., "probabilities":{...}, "noul":P(true)}}}
#
# 用法: bash serve.sh [port]   (默认 8000)
# 依赖: 先跑 install.sh (venv 在 ~/metask-jev/.venv, 模型已缓存)
set -euo pipefail
PORT="${1:-8000}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$HOME/metask-jev/.venv" ]; then
  echo "run install.sh first"; exit 1
fi
source "$HOME/metask-jev/.venv/bin/activate"
pip install -q flask 2>/dev/null || true

python "$DIR/serve.py" --port "$PORT" --model "$HOME/metask-jev/model_path.txt" 2>&1 \
  | while read -r line; do echo "$(date +%H:%M:%S) $line"; done