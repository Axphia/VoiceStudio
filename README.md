# VoiceStudio

Zero-shot voice cloning from video files with Coqui XTTS-v2. No training needed — extract audio from a video, clean it, and clone.

Tested on Windows + Python 3.10 + RTX 3050 Laptop (CUDA 12.1).

## Easy start (no commands)

Double-click **`start.bat`** — the server starts and your browser opens at http://127.0.0.1:8000. Pick a voice, type, press Clone, listen. (Backend is `scripts/api.py` + the `web/` UI.)

**New voice from a mix of files:** in the second card, type a name, select many audio/video files, press *Upload & build voice* — they are cleaned, mixed into one pooled voice, and auto-selected for cloning. Same as `ingest.py`, no commands.

**Train your own model:** the third card runs `prepare_ft` + `train_xtts` as a background job — pick source folder, language, epochs, profile (or smoke test), press Start, watch the live log. Only one job at a time. System line up top shows VRAM + active profile.

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
engines/            # pluggable TTS backends (ElevenLabs-style)
  base.py           # VoiceEngine interface
  registry.py       # discovery (built-in + entry-points)
  xtts.py           # built-in Coqui XTTS-v2 engine
scripts/
  extract_audio.py   # video -> mono WAV
  clean_ref.py       # trim + denoise + normalize ref (--denoise 0.3 expressive / 0.6 noisy)
  clone.py           # XTTS-v2 zero-shot cloning (single shot)
  compute_latents.py # speaker conditioning -> cached .pt (run once per voice)
  clone_fast.py      # model loads once + cached latents, batch --texts file, --engine
  engine.py          # list / add / test engines
  ingest.py          # batch: many files/folder -> cleaned dataset + pooled voice
  api.py             # web backend (JSON API + serves web/)
  server.py          # Gradio UI, model stays loaded (fastest for repeats)
```
Web UI lives in `web/index.html` (vanilla HTML/JS, no build step).

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

## Train from many files (batch ingest)

Point at a folder or several files — videos and audios mix freely. Each file is extracted, trimmed/denoised, logged to `manifest.json`, and concatenated into `pooled.wav` (capped at ~30s, XTTS's limit):

```powershell
uv run python scripts/ingest.py --input data/videos --out-dir data/voices/my_show
uv run python scripts/ingest.py --input a.mp4 --input b.m4a --out-dir data/voices/mix --start 2.0 --dur 8.0 --denoise 0.4
```

Then build one voice from all of them:

```powershell
uv run python scripts/compute_latents.py --speaker-wav data/voices/my_show/pooled.wav --out data/voices/my_show/voice.pt
uv run python scripts/clone_fast.py --latents data/voices/my_show/voice.pt --text "こんにちは！" --out outputs/x.wav --language ja
```

Tip: ingest one speaker/show per folder. Mixing different speakers in one pool blurs the voice.

## Train your own model (fine-tune XTTS-v2)

Zero-shot above borrows the base voice. Fine-tuning bakes YOUR voice into the weights:

```powershell
# 1. dataset: transcribe + chunk to <=11s (3+ min of speech ideally)
uv run python scripts/prepare_ft.py --input data/audio_clean --out data/ft/myvoice --language ja

# 2. smoke test: overfits 1 batch, proves it fits your VRAM
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --smoke

# 3. full run (RTX 3050 4GB profile: batch 1, grad-accum 16, fp32)
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --epochs 20

# resume: --resume training/run/<run-dir>/checkpoint_N.pth
```

PC lags while training? Train in installments — N epochs, GPU rests, auto-resume:

```powershell
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --epochs 6 --burst-epochs 2 --rest-seconds 90
```

Same checkbox exists in the web Train card. Each burst saves checkpoints and the next
resumes from the latest (`Model restored from step N`); rest gaps leave the GPU idle.

Laggy machine? Check + auto-tune first (mid profile fits RTX 3050 4GB):

```powershell
uv run python scripts/hw.py                    # CPU/RAM/VRAM report + profile
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --profile mid ...
```

Training needs ~3.8 GB free VRAM — close browser/games first or it OOMs.

Verified on RTX 3050 4GB: real epoch, loss 2.04 → 2.00, checkpoints saved, no OOM.
Notes: fp32 is default (fp16 AMP overflows to NaN on this stack — `--fp16` to try it);
checkpoints are ~5.6 GB each (`training/` is git-ignored); base weights/DVAE auto-resolve
(local XTTS download + HuggingFace `coqui/XTTS-v2`); dead Coqui-gateway URLs are worked around
(official mel_stats extracted from `model.pth`).

## Export model (.pth) + calibrate voice

Pick an epoch, pack it as a portable bundle, tune pitch/expressiveness:

```powershell
# --checkpoint = epoch pick (checkpoint_3 vs _9 = different epochs)
uv run python scripts/export_model.py --checkpoint training/run/<run>/checkpoint_9.pth --out models/myvoice

uv run python scripts/calibrate.py --text "こんにちは！" --voice data/audio_clean/deer_expressive.wav --model models/myvoice --pitch 2 --temperature 0.85 --language ja --out outputs/cali.wav
```

- `--pitch` semitones (-12..+12), `--temperature` 0.6 flat → 0.85 expressive, `--speed`, `--model` bundle dir (omit = base XTTS).
- Honest limits: the bundle is Coqui-XTTS format (loads anywhere `TTS` runs) — RVC/voice-changer apps can't read XTTS weights (different architecture), and full XTTS→ONNX isn't practical (autoregressive GPT). For ONNX changer compatibility you'd train an RVC model instead — ask if you want that pipeline.

## Fast path (optimized, RTX 3050)

`clone.py` reloads the 2 GB model on every call (~20s). For repeats, cache latents and keep the model loaded:

```powershell
# once per voice (~2s, saved to data/audio_clean/deer.pt)
uv run python scripts/compute_latents.py --speaker-wav "data/audio_clean/deer_expressive.wav" --out "data/audio_clean/deer.pt"

# batch: model loads once, then ~2s per line (texts.txt = one sentence per line)
uv run python scripts/clone_fast.py --latents "data/audio_clean/deer.pt" --texts texts.txt --out-dir "outputs/batch" --language ja --temperature 0.75

# or interactive server (zero reload between clones)
uv run python scripts/server.py
```

`--temperature`: 0.6 flatter/safer, 0.75 default, 0.85 more expressive.

## Engines (ElevenLabs-style, GitHub-importable)

`xtts` is built in. Other backends plug in via the `VoiceEngine` interface and auto-register:

```powershell
uv run python scripts/engine.py list
uv run python scripts/engine.py add git+https://github.com/user/their-engine.git
uv run python scripts/engine.py test --engine xtts --text "hello" --ref data/audio_clean/deer_expressive.wav --language en
```

Use any registered engine in batch or UI:

```powershell
uv run python scripts/clone_fast.py --engine their-engine --ref-audio data/audio_clean/deer_expressive.wav --text "hello" --out outputs/x.wav --language en
```

To publish an engine on GitHub, match the interface (no VoiceStudio dependency needed — duck-typed):

```python
# their_package/__init__.py
from pathlib import Path
class F5Engine:
    name = "f5"; label = "F5-TTS (GitHub)"; needs_ref = True
    def synthesize(self, text, out, language="en", ref_audio=None, **kw):
        ...  # write WAV to Path(out), return it
```

```toml
# their pyproject.toml
[project.entry-points."voicestudio.engines"]
f5 = "their_package:F5Engine"
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
