# 本地语音转文字工具 — 实施计划

## 1. 目标与约束

- **硬件**：MacBook M4 / 24GB 统一内存
- **环境**：conda `common`
- **输入**：1–2 小时的 MP3 文件
- **准确率目标**：WER ≤ 8%（≥92%），争取 ≤ 5%（≥95%）
- **耗时**：宽松 → 优先精度
- **语种**：中英混讲
- **输出**：TXT + SRT（顺带 VTT + JSON）
- **说话人分离**：需要（一般为 2 人对话）
- **交互方式**：CLI + macOS 拖拽 / 双击脚本

## 2. 技术选型

因为要同时做"转写 + 词级时间戳 + 说话人分离 + 合并"，主方案选 **WhisperX**：

- 内部组合 `faster-whisper`（转写） + `wav2vec2`（强制对齐） + `pyannote.audio`（说话人分离）
- 合并逻辑（词 → 说话人）官方实现稳定，不用自己写
- M4 上用 CPU + int8 跑 `large-v3`，非实时场景下耗时可接受

### 选型对比

| 方案 | 精度 | M4 速度 | 易用性 | 是否带 diarization | 备注 |
|---|---|---|---|---|---|
| `openai-whisper` (PyTorch) | 高 | 慢 | 一般 | 否 | 参考基线，不选 |
| `mlx-whisper` | 高 | **最快** | 好 | 否 | 需自己做分离 + 对齐合并 |
| `faster-whisper` | 高 | 快 | 好 | 否 | WhisperX 的底层 |
| **WhisperX** | 高 + 对齐 + 分离 | 中 | 较好 | **是** | **主方案** |
| `whisper.cpp` + Core ML | 高 | 很快 | 一般 | 否 | 配置繁琐 |

**备选**：若 WhisperX 速度不满意 → `mlx-whisper`（转写） + `pyannote.audio`（分离） + 自写合并脚本。

## 3. 模型与参数策略（冲 95%）

- **转写模型**：`large-v3`（中英混讲选它；`large-v3-turbo` 在中文上偶有退化，先稳后快）
- **compute_type**：`int8`（M4 上 faster-whisper 走 CPU int8 最稳）
- **language**：显式设 `zh`（中英混讲时 zh 模式英文保留通常更好，需 A/B）
- **task**：`transcribe`（明确告知不要翻译）
- **beam_size**：5
- **condition_on_previous_text**：`False`（长音频防漂移 / 重复，关键）
- **VAD 预分段**：WhisperX 自带，开启
- **initial_prompt**：注入 `glossary.txt`（人名 / 产品名 / 术语）
- **强制对齐**：中文段用 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`；英文段用 WhisperX 默认 `WAV2VEC2_ASR_LARGE_LV60K_960H`。若混讲比例复杂，可只用中文对齐或 `--no-align` 降级为段级时间戳
- **Diarization**：`pyannote/speaker-diarization-3.1`，已知两人时明确 `min_speakers=2, max_speakers=2`（精度最佳）

## 4. 依赖与环境
依赖conda 里面的 common环境

```
ffmpeg (conda-forge)
python 3.11
torch (Apple Silicon 官方轮子，MPS)
whisperx                # 内含 faster-whisper
pyannote.audio          # WhisperX 依赖
jiwer                   # 可选，做 WER 基准
tqdm                    # 进度条
pyyaml                  # 读 config
python-dotenv           # 读 HF_TOKEN
```

**一次性手动步骤**（`setup.sh` 会提示）：

1. HuggingFace 注册并生成 Access Token
2. 登录后访问并接受下列两个模型的使用协议：
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
3. 把 Token 写入项目根目录 `.env`：`HF_TOKEN=hf_xxx`

## 5. 项目结构

```
voice2text/
├── README.md
├── plan.md                    # 本文档
├── environment.yml            # conda 环境
├── config.yaml                # 默认配置（模型、语言、说话人数等）
├── .env.example               # HF_TOKEN 占位
├── prompts/
│   └── glossary.txt           # 专有名词 / 人名，可随时编辑
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # transcribe + align + diarize 主流程
│   ├── preprocess.py          # ffmpeg 归一化为 16k mono wav
│   ├── postprocess.py         # 合并同说话人 + SRT/TXT/VTT/JSON 生成
│   └── cli.py                 # argparse 入口
├── scripts/
│   ├── transcribe.command     # macOS 双击 / 拖拽入口
│   └── setup.sh               # 一次性环境安装脚本
├── input/                     # 放 mp3
└── output/                    # 生成结果
    └── <filename>/
        ├── transcript.txt
        ├── transcript.srt
        ├── transcript.vtt
        └── transcript.json
```

## 6. CLI 设计

```bash
# 单文件
python -m src.cli input/meeting.mp3

# 批量 + 指定说话人数
python -m src.cli input/ --speakers 2

# 常用可选参数
--model large-v3            # 默认
--language zh               # 默认
--speakers 2                # 已知 2 人就写 2，否则 auto
--no-diarize                # 不要说话人分离
--no-align                  # 跳过强制对齐（更快，时间戳粗一点）
--glossary prompts/glossary.txt
--output-dir output/
```

## 7. macOS 拖拽脚本

`scripts/transcribe.command`（`chmod +x` 后，Finder 里双击或把 mp3 拖到图标上）：

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate common

if [ $# -eq 0 ]; then
  # 双击时没参数 → 默认处理 input/ 目录
  python -m src.cli input/
else
  python -m src.cli "$@"
fi
echo "完成。按任意键关闭..."
read -n 1
```

## 8. 输出样例

**TXT**（合并同说话人后按段）：

```
[00:00:03] S1: 我们今天讨论一下 roadmap 的事情...
[00:00:11] S2: OK，我先过一下 Q3 的 milestones...
```

**SRT**（细粒度 + 说话人前缀）：

```
1
00:00:03,120 --> 00:00:07,400
[S1] 我们今天讨论一下 roadmap 的事情

2
00:00:07,400 --> 00:00:11,050
[S1] 主要有三个点需要 align
```

## 9. 实施步骤

1. **环境搭建**
   - 写 `environment.yml`（Python 3.11 + ffmpeg + pytorch + whisperx）
   - 写 `setup.sh`：创建 / 更新 conda env、提示 HF 协议、写 `.env`
2. **核心 pipeline**
   - `preprocess.py`：ffmpeg → 16kHz mono wav（临时文件）
   - `pipeline.py`：`whisperx.load_model` → `transcribe` → `load_align_model` → `align` → `DiarizationPipeline` → `assign_word_speakers`
   - `postprocess.py`：合并连续同说话人 segment；生成 TXT/SRT/VTT/JSON
3. **CLI + 批处理 + 进度条**（tqdm）
4. **拖拽脚本 + README**
   - 写清 HF Token 配置
   - 说明首次运行会下载约 4–5GB 模型（`large-v3` ≈ 3GB + 对齐 + 分离模型）
5. **基准验证**
   - 1 段样本肉眼抽检 + `jiwer` 计算 WER
   - 不达标 → 切 `--language en` / 换 `large-v3-turbo` / 调 `beam_size`

## 10. 预期性能（M4 24GB）

| 阶段 | 1h 音频耗时 |
|---|---|
| ffmpeg 预处理 | < 30s |
| faster-whisper large-v3 (int8) | 20–40 min |
| wav2vec2 对齐 | 3–5 min |
| pyannote diarization 3.1 | 5–10 min |
| **总计** | **约 30–60 min** |

内存峰值 ≤ 8GB。

## 11. 风险与兜底

- **HF Token / 模型协议**：首次必须手动接受协议，否则 diarization 启动失败。`setup.sh` 给出清晰报错 + 链接。
- **中英混讲时英文被翻译**：显式 `task="transcribe"`；glossary 内保留若干英文样例词。
- **两人声音相近导致分离合并错误**：明确 `min_speakers=max_speakers=2`；实在不行用 `--no-diarize` 退化为无说话人版本。
- **MPS / Metal 偶发崩溃**：默认 CPU + int8，不依赖 MPS，稳定性优先。
- **超长文件 OOM**：WhisperX 的 VAD 会自动切片；若仍 OOM，降 `batch_size` 或切 `large-v3-turbo`。
- **WhisperX 速度不满意**：切备选 `mlx-whisper` + `pyannote.audio` 自写合并。

## 12. 待确认 / 可调项

- `large-v3`（稳）vs `large-v3-turbo`（快 3–5 倍）：默认 `large-v3`，后续基准后再决定
- 是否需要繁简统一 / 英文术语大小写后处理表（MVP 不做）
- 是否需要 GUI（后续再议，MVP 只做 CLI + `.command`）
