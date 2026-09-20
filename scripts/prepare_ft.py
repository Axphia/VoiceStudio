"""Prepare an XTTS fine-tune dataset from WAVs.

Transcribes with faster-whisper (CPU, since ctranslate2 lacks CUDA libs here),
cuts speech into <=11s chunks (XTTS limit ~11.6s), writes:

  <out>/wavs/*.wav
  <out>/metadata.csv      # wavs/xxx.wav|text|text  (ljspeech format)

Usage:
  uv run python scripts/prepare_ft.py --input data/audio_clean --out data/ft/deer --language ja
  uv run python scripts/prepare_ft.py --input data/voices/my_show/clean --out data/ft/my_show --language ja --speaker myshow
"""
import argparse
import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import librosa
import soundfile as sf

SR = 22050
MAX_SEG = 11.0


def transcribe(wav: Path, language: str):
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav), language=language or None)
    detected = info.language
    return list(segments), detected


def prepare(input_dir: Path, out_dir: Path, language: str, speaker: str):
    wavs = sorted([p for p in Path(input_dir).glob("*.wav")])
    if not wavs:
        raise SystemExit(f"No .wav in {input_dir}")
    wout = out_dir / "wavs"
    wout.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0.0
    for wav in wavs:
        segs, detected = transcribe(wav, language)
        lang = language or detected
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
        for s in segs:
            dur = s.end - s.start
            text = s.text.strip()
            if dur < 1.0 or not text:
                continue
            # split long segments in half until each <= MAX_SEG
            spans = [(s.start, s.end)]
            while any(e - b > MAX_SEG for b, e in spans):
                b, e = spans.pop(0)
                m = (b + e) / 2
                spans += [(b, m), (m, e)]
            for b, e in spans:
                clip = y[int(b * SR):int(e * SR)]
                if len(clip) / SR < 1.0:
                    continue
                name = f"{wav.stem}_{len(rows):04d}"
                sf.write(str(wout / f"{name}.wav"), clip, SR)
                rows.append((name, text))  # bare stem: ljspeech fmt adds wavs/+.wav
                total += len(clip) / SR

    meta = out_dir / "metadata.csv"
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerows((a, t, t) for a, t in rows)

    print(f"[prepare_ft] {len(rows)} clips, {total/60:.1f} min, lang={language}")
    print(f"[prepare_ft] -> {meta}")
    if total < 60:
        print("[prepare_ft] WARNING: <1 min of speech. XTTS fine-tune wants "
              "3+ min (ideally 10+). Add more files via ingest.py.")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="folder of .wav (e.g. ingest clean/)")
    p.add_argument("--out", required=True, help="e.g. data/ft/deer")
    p.add_argument("--language", default="ja")
    p.add_argument("--speaker", default="ft_voice")
    a = p.parse_args()
    prepare(Path(a.input), Path(a.out), a.language, a.speaker)
