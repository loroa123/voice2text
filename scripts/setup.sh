#!/usr/bin/env bash
# 一次性环境安装脚本。
# 使用方式：bash scripts/setup.sh
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 未检测到 conda，请先安装 Miniconda / Anaconda。" >&2
  exit 1
fi

ENV_NAME="${CONDA_ENV_NAME:-common}"
echo "[info] 目标 conda 环境：${ENV_NAME}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[info] 环境已存在，使用 env update 增量安装。"
  conda env update -n "${ENV_NAME}" -f environment.yml
else
  echo "[info] 创建新环境。"
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

conda activate "${ENV_NAME}"

if [ ! -f "${ROOT}/.env" ]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
  echo "[info] 已创建 .env，请编辑填入 HF_TOKEN。"
fi

cat <<'EOF'

==============================================================
完成环境安装。接下来还需要 3 步：

1. 到 https://huggingface.co/settings/tokens 生成一个 Read Token。
2. 登录后依次访问并点击 "Agree and access repository"：
     https://huggingface.co/pyannote/segmentation-3.0
     https://huggingface.co/pyannote/speaker-diarization-3.1
3. 编辑当前目录的 .env，把 HF_TOKEN=hf_xxx 替换为真实 Token。

完成后：
  - 把 mp3 放入 input/ 目录
  - 双击 scripts/transcribe.command，或
  - 终端执行：
      conda activate common
      python -m src.cli input/
==============================================================
EOF
