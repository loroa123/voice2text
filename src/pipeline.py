"""WhisperX pipeline：转写 → 强制对齐 → 说话人分离。"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    model_name: str = "large-v3"
    compute_type: str = "int8"
    device: str = "cpu"
    batch_size: int = 8
    language: str = "zh"
    task: str = "transcribe"
    beam_size: int = 5
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    align_enabled: bool = True
    align_zh_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    diarize_enabled: bool = True
    min_speakers: int = 2
    max_speakers: int = 2
    initial_prompt: str | None = None
    hf_token: str | None = None


def _load_glossary_prompt(glossary_path: Path | None) -> str | None:
    if glossary_path is None:
        return None
    p = Path(glossary_path)
    if not p.exists():
        return None
    words: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    if not words:
        return None
    return "涉及术语：" + "、".join(words) + "。"


def transcribe_file(audio_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    """对单个 16k mono wav 执行完整流程，返回带 speaker 标注的 segments 字典。"""
    import whisperx  # 延迟导入，避免 CLI --help 时也加载大库

    audio_path = Path(audio_path)
    audio = whisperx.load_audio(str(audio_path))

    asr_options: dict[str, Any] = {
        "beam_size": cfg.beam_size,
        "condition_on_previous_text": cfg.condition_on_previous_text,
    }
    if cfg.initial_prompt:
        asr_options["initial_prompt"] = cfg.initial_prompt

    model = whisperx.load_model(
        cfg.model_name,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
        task=cfg.task,
        asr_options=asr_options,
    )
    result = model.transcribe(audio, batch_size=cfg.batch_size)
    # language 字段下游对齐会用到
    result.setdefault("language", cfg.language)

    # 主动释放 ASR 模型
    del model
    gc.collect()

    # 强制对齐（词级时间戳）
    if cfg.align_enabled:
        try:
            align_kwargs: dict[str, Any] = {
                "language_code": result["language"],
                "device": cfg.device,
            }
            if result["language"].startswith("zh"):
                align_kwargs["model_name"] = cfg.align_zh_model
            align_model, metadata = whisperx.load_align_model(**align_kwargs)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                cfg.device,
                return_char_alignments=False,
            )
            del align_model
            gc.collect()
        except Exception as e:  # 对齐失败不致命，继续走段级时间戳
            print(f"[warn] align 失败，降级为段级时间戳：{e}")

    # 说话人分离
    if cfg.diarize_enabled:
        if not cfg.hf_token:
            raise RuntimeError(
                "需要 HF_TOKEN 才能做说话人分离。请在 .env 设置 HF_TOKEN，或使用 --no-diarize。"
            )
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=cfg.hf_token,
            device=cfg.device,
        )
        diarize_segments = diarize_pipeline(
            str(audio_path),
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        del diarize_pipeline
        gc.collect()

    return result


def build_config(raw: dict[str, Any], overrides: dict[str, Any]) -> PipelineConfig:
    model = raw.get("model", {})
    trans = raw.get("transcribe", {})
    align = raw.get("align", {})
    diar = raw.get("diarize", {})

    cfg = PipelineConfig(
        model_name=model.get("name", "large-v3"),
        compute_type=model.get("compute_type", "int8"),
        device=model.get("device", "cpu"),
        batch_size=int(model.get("batch_size", 8)),
        language=trans.get("language", "zh"),
        task=trans.get("task", "transcribe"),
        beam_size=int(trans.get("beam_size", 5)),
        condition_on_previous_text=bool(trans.get("condition_on_previous_text", False)),
        vad_filter=bool(trans.get("vad_filter", True)),
        align_enabled=bool(align.get("enabled", True)),
        align_zh_model=align.get(
            "zh_model", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        ),
        diarize_enabled=bool(diar.get("enabled", True)),
        min_speakers=int(diar.get("min_speakers", 2)),
        max_speakers=int(diar.get("max_speakers", 2)),
        hf_token=os.environ.get("HF_TOKEN"),
    )

    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg
