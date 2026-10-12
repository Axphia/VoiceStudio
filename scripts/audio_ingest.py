"""Batch ingest: many audio/video files (or a folder) -> cleaned voice dataset.

Handles what extract_audio.py does for one file, for N files:
video --ffmpeg--> wav --trim/denoise--> clean wav, plus an optional
pooled reference (best segments concatenated, capped at ~30s for XTTS).

Usage:
  # whole folder
  uv run python scripts/ingest.py --input data/videos --out-dir data/voices/deer_show
  # several files
  uv run python scripts/ingest.py --input a.mp4 --input b.m4a --out-dir data/voices/mix
  # then build the voice:
  uv run python scripts/compute_latents.py --speaker-wav data/voices/deer_show/pooled.wav --out data/voices/deer_show/voice.pt
"""
import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import librosa
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # project root (engines package)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"cannot load {name}.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*"))
                      if f.is_file() and f.suffix.lower() in VIDEO_EXTS | AUDIO_EXTS]
        elif p.is_file():
            files.append(p)
        else:
            print(f"[ingest] skip (not found): {raw}")
    # de-dupe, keep order
    return list(dict.fromkeys(files))


def ingest(inputs, out_dir, sr=22050, start=0.0, dur=10.0, denoise=0.5,
           pool=True, pool_max=30.0):
    extract_mod = _load_sibling("extract_audio")
    clean_mod = _load_sibling("clean_ref")

    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    pooled_parts: list[np.ndarray] = []
    pooled_len = 0.0

    for src in collect_inputs(inputs):
        stem = src.stem[:60]
        print(f"[ingest] {src.name}")
        try:
            if src.suffix.lower() in VIDEO_EXTS:
                wav = raw_dir / f"{stem}.wav"
                if wav.exists():  # avoid collisions on truncated names
                    i = 1
                    while (raw_dir / f"{stem}_{i}.wav").exists():
                        i += 1
                    stem = f"{stem}_{i}"
                    wav = raw_dir / f"{stem}.wav"
                extract_mod.extract_audio(src, wav, sr)
            else:
                y, _ = librosa.load(str(src), sr=sr, mono=True)
                wav = raw_dir / f"{stem}.wav"
                sf.write(str(wav), y, sr)

            cleaned = clean_dir / f"{stem}.wav"
            clean_mod.clean(wav, cleaned, start, dur, sr, denoise)
            d = librosa.get_duration(path=str(cleaned))
            manifest.append({"source": str(src), "raw": str(wav),
                             "clean": str(cleaned), "seconds": round(d, 2)})

            if pool and pooled_len < pool_max:
                y, _ = librosa.load(str(cleaned), sr=sr, mono=True)
                need = int((pool_max - pooled_len) * sr)
                y = y[:need]
                pooled_parts.append(y)
                pooled_len += len(y) / sr
        except Exception as e:
            print(f"[ingest] FAILED {src.name}: {e}")
            manifest.append({"source": str(src), "error": str(e)})

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    pooled_path = None
    if pool and pooled_parts:
        pooled = np.concatenate(pooled_parts)
        pooled = pooled / (np.abs(pooled).max() + 1e-9) * 0.7
        pooled_path = out_dir / "pooled.wav"
        sf.write(str(pooled_path), pooled, sr)
        print(f"[ingest] pooled {len(pooled)/sr:.1f}s from "
              f"{len(pooled_parts)} files -> {pooled_path}")

    ok = sum(1 for m in manifest if "clean" in m)
    print(f"[ingest] done: {ok}/{len(manifest)} ok -> {out_dir}")
    return pooled_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="file or folder; repeatable")
    p.add_argument("--out-dir", required=True, help="e.g. data/voices/my_show")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--dur", type=float, default=10.0)
    p.add_argument("--denoise", type=float, default=0.5)
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--pool-max", type=float, default=30.0)
    a = p.parse_args()
    ingest(a.input, Path(a.out_dir), a.sr, a.start, a.dur,
           a.denoise, not a.no_pool, a.pool_max)
