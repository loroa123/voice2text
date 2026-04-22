"""音频预处理：统一转成 16kHz 单声道 WAV，喂给 WhisperX 最稳。"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".mov"}


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "未找到 ffmpeg。请先安装：conda install -c conda-forge ffmpeg"
        )


def to_wav_16k_mono(src: Path, dst_dir: Path | None = None) -> Path:
    """将任意音频/视频转为 16kHz 单声道 PCM WAV，返回输出路径。"""
    ensure_ffmpeg()
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    if dst_dir is None:
        dst_dir = Path(tempfile.mkdtemp(prefix="voice2text_"))
    else:
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / (src.stem + ".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-f", "wav",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def collect_inputs(path: Path) -> list[Path]:
    """接受文件或目录；目录则递归收集支持的音视频文件。"""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
    raise FileNotFoundError(path)
