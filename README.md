# VoiceStudio

Zero-shot voice cloning from video files with Coqui XTTS-v2. No training needed — extract audio from a video, clean it, and clone.

Tested on Windows + Python 3.10 + RTX 3050 Laptop (CUDA 12.1).

## Setup (uv)

```powershell
uv sync
```

Notes:
- `torch` / `torchaudio` come from the PyTorch CUDA 12.1 index (see `pyproject.toml`).
- First XTTS run downloads ~1.9 GB to `%LOCALAPPDATA%\tts`.
- On first download, answer `y` to the Coqui CPML prompt (non-commercial use).
- Pins that matter: `transformers==4.46.3` (v5 breaks TTS), `onnxruntime==1.23.2` (py310), `setuptools<81` (librosa `pkg_resources`), `cutlet/fugashi/unidic-lite` (Japanese).

## Project layout

```text
data/videos/        # input .mp4 files
data/audio_clean/   # extracted / cleaned reference WAVs
outputs/            # cloned WAVs
scripts/
  extract_audio.py  # video -> mono WAV
  clean_ref.py      # trim + denoise + normalize ref
  clone.py          # XTTS-v2 zero-shot cloning
```

## Usage

1. Extract audio from video:

```powershell
uv run python scripts/extract_audio.py --video "data/videos/YOUR_VIDEO.mp4" --out "data/audio_clean/ref.wav"
```

2. Clean reference (6–10s clean single-speaker is ideal):

```powershell
uv run python scripts/clean_ref.py --in "data/audio_clean/ref.wav" --out "data/audio_clean/clean.wav" --start 3.0 --dur 8.0
```

3. Clone:

```powershell
uv run python scripts/clone.py --text "Hello world" --speaker-wav "data/audio_clean/clean.wav" --out "outputs/cloned.wav" --language en
```

Options: `--language en|ja|...`, `--speed 0.95` (slower, more natural), `--no-split` (disable sentence splitting).

## Worked example (repo files)

Both sample videos are Japanese, so use `--language ja` for same-language cloning:

```powershell
uv run python scripts/extract_audio.py --video "data/videos/The judgmental deer has a lot to say 🦌 #deer #japanese #memes  [7680228558031146261].mp4" --out "data/audio_clean/deer_ref.wav"

uv run python scripts/clone.py --text "こんにちは、これはテストです。" --speaker-wav "data/audio_clean/deer_clean.wav" --out "outputs/deer_ja.wav" --language ja
```

Expressive (more intonation) — use punctuation and an emotional ref segment:

```powershell
uv run python scripts/clone.py --text "えっ、本当に？信じられないよ！すごく嬉しいなぁ…" --speaker-wav "data/audio_clean/deer_expressive.wav" --out "outputs/deer_expressive_ja.wav" --language ja --speed 0.95
```

## Tips for natural voice (น้ำเสียง)

- Reference: 6–12s, one speaker, no music/SFX. Full 17s noisy clips sound robotic.
- Match language: JA speaker + `--language en` gives cross-lingual accent. Same-language first to check quality.
- Denoise lightly (`prop_decrease` 0.3 for expressive, 0.6 for noisy). Heavy denoise flattens dynamics.
- Write text with `? ! … 、` and short sentences; `--speed 0.9–0.95` helps.
- XTTS-v2 has no Thai. For Thai output you need a different model (e.g. F5-TTS / OpenVoice).

## Troubleshooting

- `EOFError` on first run → pipe `y`: `"y" | uv run python scripts/clone.py ...`
- `BeamSearchScorer` import error → you have transformers v5; `uv sync` pins 4.46.3.
- `pkg_resources` missing → needs `setuptools<81`; `uv sync` handles it.
- `No module named 'cutlet'` → needed for Japanese; installed via `pyproject.toml`.
- faster-whisper `cublas64_12.dll` error on CUDA → fall back to `WhisperModel(..., device="cpu", compute_type="int8")`.
