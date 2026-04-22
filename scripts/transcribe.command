#!/usr/bin/env bash
# macOS 拖拽 / 双击入口。
# 用法：
#   - 双击：处理 input/ 目录
#   - 把 mp3 拖到本文件图标：处理所有拖入文件
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 未检测到 conda，请先运行 scripts/setup.sh。" >&2
  read -n 1 -s -r -p "按任意键关闭..."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV_NAME:-common}"
conda activate "${ENV_NAME}"

if [ $# -eq 0 ]; then
  python -m src.cli input/
else
  python -m src.cli "$@"
fi

echo
read -n 1 -s -r -p "处理完成，按任意键关闭..."
echo
