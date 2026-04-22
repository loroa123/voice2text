# voice2text

macOS Apple Silicon 上的本地语音转文字小工具。基于 [WhisperX](https://github.com/m-bain/whisperx)：Whisper 转写 + wav2vec2 强制对齐 + pyannote 说话人分离。

- 语种：中英混讲优化（可切 `en`）
- 输出：TXT / SRT / VTT / JSON
- 说话人分离：默认开启，按 `S1 / S2` 标注
- 适合：1–2 小时长音频，精度优先（`large-v3`）

## 一、快速开始

```bash
# 1) 克隆后进入目录
cd voice2text

# 2) 一键安装（基于已有 conda 环境名 common）
bash scripts/setup.sh

# 3) 配置 HuggingFace Token（说话人分离必需）
#    先到 https://huggingface.co/settings/tokens 生成 Read Token
#    再到下面页面点 "Agree and access repository"（默认 diarization 模型）：
#      https://huggingface.co/pyannote/segmentation-3.0
#      https://huggingface.co/pyannote/speaker-diarization-3.1
#    然后编辑项目根目录的 .env：
#      HF_TOKEN=hf_xxx

# 4) 把 mp3 放进 input/ 目录，然后：
#    方式 A - 双击脚本（或把文件拖到图标上）
open scripts/transcribe.command

#    方式 B - 命令行
conda activate common
python -m src.cli input/
```

首次运行会自动下载模型（约 4–5GB），缓存在 `~/.cache/huggingface`。

## 二、命令行用法

```bash
# 单个文件
python -m src.cli input/meeting.mp3

# 目录（递归）
python -m src.cli input/

# 已知两人对话
python -m src.cli input/meeting.mp3 --speakers 2

# 跳过说话人分离（最快）
python -m src.cli input/meeting.mp3 --no-diarize

# 切到英文模型（纯英文音频）
python -m src.cli input/meeting.mp3 --language en

# 试更快的 turbo
python -m src.cli input/meeting.mp3 --model large-v3-turbo
```

主要参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model` | Whisper 模型名 | `large-v3` |
| `--language` | 语言代码 (`zh`/`en`) | `zh` |
| `--speakers` | 已知说话人数，同时设 min/max | 无 |
| `--min-speakers` / `--max-speakers` | 分别控制 | `2 / 2` |
| `--no-diarize` | 跳过说话人分离 | 关 |
| `--no-align` | 跳过强制对齐（快，时间戳粗） | 关 |
| `--compute-type` | `int8` / `float16` / `float32` | `int8` |
| `--device` | `cpu` / `mps` / `cuda` | `cpu` |
| `--glossary` | 术语表路径 | `prompts/glossary.txt` |
| `--output-dir` | 输出目录 | `output/` |
| `--keep-wav` | 保留中间 WAV | 关 |

`config.yaml` 中可配置说话人分离模型，默认使用：

`pyannote/speaker-diarization-3.1`

如果你改用 `pyannote/speaker-diarization-community-1`，需额外到对应 HF 页面申请访问权限。

## 三、术语表（提升识别率）

编辑 `prompts/glossary.txt`，每行一个词（中英文混排均可）。运行时会作为 `initial_prompt` 注入模型，显著提升人名、产品名、英文术语的识别率。

示例：

```
roadmap
milestone
Q3 kickoff
张三
李四
```

## 四、输出示例

结果放在 `output/<文件名>/`：

- `transcript.txt` — 说话人合并后的可读稿
- `transcript.srt` — 字幕文件（含 `[S1]` 前缀）
- `transcript.vtt` — Web 字幕
- `transcript.json` — 原始词级 / 说话人数据

TXT 片段示例：

```
[00:00:03] S1: 我们今天讨论一下 roadmap 的事情
[00:00:11] S2: OK，我先过一下 Q3 的 milestones
```

## 五、性能参考（M4 24GB，CPU + int8）

| 阶段 | 1h 音频 |
|---|---|
| ffmpeg 预处理 | <30s |
| Whisper large-v3 转写 | 20–40 min |
| wav2vec2 对齐 | 3–5 min |
| pyannote 说话人分离 | 5–10 min |
| **总计** | **30–60 min** |

内存峰值 ≤ 8GB。

## 六、常见问题

- **`Could not download pyannote model`**：说明没接受模型协议或 Token 无权限。到第 1 步列出的两个页面重新点 Agree。
- **中英混讲中英文被翻译成中文**：确保 `task=transcribe`（默认）；在 `prompts/glossary.txt` 加几个英文词做示例。
- **处理超长文件卡住**：降 `--batch-size 4`，或换 `--model large-v3-turbo`。
- **识别某些专有词总是错**：加入 `prompts/glossary.txt` 后再跑一次。
- **想用 MPS 加速**：`--device mps --compute-type float16`。注意 MPS 在 faster-whisper 上支持不完美，不稳定就切回 `cpu + int8`。

## 七、目录结构

```
voice2text/
├── README.md
├── plan.md                   # 设计文档
├── environment.yml
├── requirements.txt
├── config.yaml
├── .env.example
├── prompts/glossary.txt
├── src/
│   ├── preprocess.py
│   ├── pipeline.py
│   ├── postprocess.py
│   └── cli.py
├── scripts/
│   ├── setup.sh
│   └── transcribe.command
├── input/
└── output/
```
# voice2text
