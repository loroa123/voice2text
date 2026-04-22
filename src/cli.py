"""
CLI 入口：python -m src.cli <file_or_dir> [options]
# 方式 A：双击 / 拖 mp3 到这个文件
open scripts/transcribe.command

# 方式 B：命令行
conda activate common
python -m src.cli input/your.mp3 --speakers 2
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .pipeline import PipelineConfig, build_config, transcribe_file
from .pipeline import _load_glossary_prompt  # noqa: F401 (re-export for tests)
from .postprocess import write_all
from .preprocess import collect_inputs, to_wav_16k_mono


ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice2text",
        description="本地语音转文字（WhisperX，支持说话人分离，输出 TXT/SRT/VTT/JSON）",
    )
    p.add_argument("inputs", nargs="+", help="音频文件或目录（目录会递归处理）")
    p.add_argument("--config", default=str(ROOT / "config.yaml"), help="配置文件路径")
    p.add_argument("--output-dir", default=None, help="输出根目录，默认读配置 output/")
    p.add_argument("--model", dest="model_name", default=None, help="Whisper 模型，如 large-v3 / large-v3-turbo")
    p.add_argument("--language", default=None, help="语言代码，如 zh / en")
    p.add_argument("--speakers", type=int, default=None, help="已知说话人数（同时设为 min/max）")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--no-diarize", action="store_true", help="跳过说话人分离")
    p.add_argument("--no-align", action="store_true", help="跳过强制对齐（更快，时间戳粗一点）")
    p.add_argument("--glossary", default=None, help="术语表路径，默认读配置")
    p.add_argument("--compute-type", default=None, help="int8 / float16 / float32")
    p.add_argument("--device", default=None, help="cpu / mps / cuda")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--keep-wav", action="store_true", help="保留中间 WAV 文件")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args(argv)

    raw_cfg = _load_yaml(Path(args.config))

    overrides: dict[str, Any] = {
        "model_name": args.model_name,
        "language": args.language,
        "compute_type": args.compute_type,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.speakers is not None:
        overrides["min_speakers"] = args.speakers
        overrides["max_speakers"] = args.speakers
    if args.min_speakers is not None:
        overrides["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        overrides["max_speakers"] = args.max_speakers
    if args.no_diarize:
        overrides["diarize_enabled"] = False
    if args.no_align:
        overrides["align_enabled"] = False

    cfg: PipelineConfig = build_config(raw_cfg, overrides)

    glossary_path = Path(args.glossary or raw_cfg.get("glossary_path") or ROOT / "prompts/glossary.txt")
    cfg.initial_prompt = _load_glossary_prompt(glossary_path)

    out_root = Path(args.output_dir or raw_cfg.get("output_dir") or "output")
    out_root = (ROOT / out_root) if not out_root.is_absolute() else out_root
    out_root.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for item in args.inputs:
        files.extend(collect_inputs(Path(item)))
    if not files:
        print("未找到可处理的音频文件。", file=sys.stderr)
        return 2

    print(f"[info] 模型: {cfg.model_name} | 语言: {cfg.language} | diarize: {cfg.diarize_enabled} | align: {cfg.align_enabled}")
    if cfg.diarize_enabled and not cfg.hf_token:
        print("[error] 开启了说话人分离但未找到 HF_TOKEN。请在 .env 填入或加 --no-diarize。", file=sys.stderr)
        return 3

    total = len(files)
    failures: list[tuple[Path, str]] = []
    for idx, audio in enumerate(files, start=1):
        print(f"\n[{idx}/{total}] >>> {audio}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="voice2text_"))
        t0 = time.time()
        try:
            wav = to_wav_16k_mono(audio, dst_dir=tmp_dir)
            print(f"  预处理完成: {wav.name}")

            result = transcribe_file(wav, cfg)

            case_out = out_root / audio.stem
            paths = write_all(result, case_out)
            elapsed = time.time() - t0
            print(f"  完成，用时 {_format_duration(elapsed)}")
            for k, p in paths.items():
                print(f"    - {k}: {p}")
        except KeyboardInterrupt:
            print("\n[info] 用户中断", file=sys.stderr)
            return 130
        except Exception as e:
            failures.append((audio, str(e)))
            print(f"  [error] 处理失败: {e}", file=sys.stderr)
        finally:
            if not args.keep_wav:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if failures:
        print("\n以下文件处理失败：", file=sys.stderr)
        for a, msg in failures:
            print(f"  - {a}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
