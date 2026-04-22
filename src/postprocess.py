"""把 WhisperX 结果写成 TXT / SRT / VTT / JSON。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _format_ts(seconds: float, comma: bool = True) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _speaker_label(raw: str | None) -> str:
    """把 pyannote 的 SPEAKER_00 / SPEAKER_01 映射成 S1 / S2。"""
    if not raw:
        return "S?"
    if raw.startswith("SPEAKER_"):
        try:
            idx = int(raw.split("_", 1)[1])
            return f"S{idx + 1}"
        except ValueError:
            return raw
    return raw


def _iter_segments(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for seg in result.get("segments", []):
        if "start" not in seg or "end" not in seg:
            continue
        yield seg


def merge_by_speaker(
    result: dict[str, Any],
    max_gap: float = 1.0,
    max_chars: int = 120,
) -> list[dict[str, Any]]:
    """合并相邻同说话人 segment，便于 TXT 阅读。"""
    merged: list[dict[str, Any]] = []
    for seg in _iter_segments(result):
        spk = _speaker_label(seg.get("speaker"))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        item = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": spk,
            "text": text,
        }
        if merged:
            last = merged[-1]
            gap = item["start"] - last["end"]
            same_speaker = last["speaker"] == spk
            short_enough = len(last["text"]) + len(text) + 1 <= max_chars
            if same_speaker and gap <= max_gap and short_enough:
                last["end"] = item["end"]
                joiner = "" if _looks_cjk(last["text"][-1:]) and _looks_cjk(text[:1]) else " "
                last["text"] = (last["text"] + joiner + text).strip()
                continue
        merged.append(item)
    return merged


def _looks_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def write_txt(segments: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for seg in segments:
        ts = _format_ts(seg["start"], comma=False).split(".")[0]
        lines.append(f"[{ts}] {seg['speaker']}: {seg['text']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(result: dict[str, Any], path: Path) -> None:
    out: list[str] = []
    for i, seg in enumerate(_iter_segments(result), start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = _speaker_label(seg.get("speaker"))
        start = _format_ts(seg["start"])
        end = _format_ts(seg["end"])
        out.append(str(i))
        out.append(f"{start} --> {end}")
        out.append(f"[{spk}] {text}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_vtt(result: dict[str, Any], path: Path) -> None:
    out: list[str] = ["WEBVTT", ""]
    for seg in _iter_segments(result):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = _speaker_label(seg.get("speaker"))
        start = _format_ts(seg["start"], comma=False)
        end = _format_ts(seg["end"], comma=False)
        out.append(f"{start} --> {end}")
        out.append(f"[{spk}] {text}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_json(result: dict[str, Any], path: Path) -> None:
    # 只保留可序列化字段
    safe_segments = []
    for seg in result.get("segments", []):
        safe_segments.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "speaker": seg.get("speaker"),
            "words": [
                {
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "word": w.get("word"),
                    "speaker": w.get("speaker"),
                    "score": w.get("score"),
                }
                for w in (seg.get("words") or [])
            ],
        })
    payload = {
        "language": result.get("language"),
        "segments": safe_segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_all(result: dict[str, Any], out_dir: Path, stem: str = "transcript") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_by_speaker(result)
    paths = {
        "txt": out_dir / f"{stem}.txt",
        "srt": out_dir / f"{stem}.srt",
        "vtt": out_dir / f"{stem}.vtt",
        "json": out_dir / f"{stem}.json",
    }
    write_txt(merged, paths["txt"])
    write_srt(result, paths["srt"])
    write_vtt(result, paths["vtt"])
    write_json(result, paths["json"])
    return paths
