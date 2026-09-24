# VoiceStudio

Zero-shot and fine-tuned voice cloning from video files with Coqui XTTS-v2. Extract audio from any video, clean it, and clone — or fine-tune your own model on Google Colab.

Tested on Windows + Python 3.10 + RTX 3050 Laptop (CUDA 12.1).

## Quick Start (no commands)

Double-click **`start.bat`** — the server starts and your browser opens at http://127.0.0.1:9999.
Pick a voice, type your text, press **Clone**, listen.

> **Note:** Port is **9999** (not 8000) to avoid Windows Hyper-V reserved port conflicts.

**New voice from files:** In the Upload card, type a name, select audio/video files, press *Upload & build voice* — they are cleaned, mixed into one pooled voice, and auto-selected for cloning.

**Train your own model (web UI):** The Train card runs `prepare_ft` + `train_xtts` as a background job — pick source folder, language, epochs, press Start, watch the live log.

## Colab Training (recommended for better quality)

For best results, train on Google Colab with a GPU. Double-click **`colab_prep.bat`** to open the Colab Tools app at http://127.0.0.1:9998.

### Tab 1 — Prepare for Colab
1. Enter voice name, language, epochs, video folder
2. Press **🚀 Run Pipeline** — auto-runs Ingest → Prepare → Pack
3. Download the generated `.ipynb` + `dataset.zip`
4. Upload both files to Google Colab and run the notebook

### Tab 2 — Install Trained Model
After training on Colab:
1. Download the checkpoint folder from Google Drive as a `.zip`
2. Upload it in the **Install Trained Model** tab
3. Enter a model name, press **📥 Install Model**
4. Restart `start.bat` — your model is ready to use

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
data/voices/        # extracted + cleaned voice folders
  <name>/
    uploads/        # original uploaded files
    clean/          # cleaned segments
    pooled.wav      # merged reference voice
data/ft/            # fine-tune datasets (prepared by prepare_ft.py)
models/             # exported fine-tuned models
outputs/            # cloned WAVs
training/           # training checkpoints (git-ignored)
engines/            # pluggable TTS backends
  base.py           # VoiceEngine interface
  registry.py       # discovery (built-in + entry-points)
  xtts.py           # built-in Coqui XTTS-v2 engine
scripts/
  api.py            # web backend (JSON API + serves web/)
  ingest.py         # batch: many files/folder → cleaned dataset + pooled voice
  prepare_ft.py     # transcribe + segment for XTTS fine-tuning
  train_xtts.py     # XTTS-v2 fine-tuning (local or Colab)
  export_model.py   # checkpoint → portable model bundle
  colab_prep.py     # Gradio app: pipeline prep + model install
  extract_audio.py  # video → mono WAV
  clean_ref.py      # trim + denoise + normalize ref
  clone.py          # XTTS-v2 zero-shot cloning (single shot)
  compute_latents.py# speaker conditioning → cached .pt (run once per voice)
  clone_fast.py     # model loads once + cached latents, batch --texts file
  hw.py             # hardware profile detection
  calibrate.py      # tune pitch / temperature on exported model
web/                # vanilla HTML/JS UI (no build step)
start.bat           # launch VoiceStudio → http://127.0.0.1:9999
colab_prep.bat      # launch Colab Tools → http://127.0.0.1:9998
colab_train_*.ipynb # auto-generated Colab training notebooks
```

## Usage (command line)

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

Options: `--language en|ja|zh|...`, `--speed 0.95` (slower, more natural), `--no-split` (disable sentence splitting).

## Batch ingest from many files

Point at a folder or several files — videos and audios mix freely:

```powershell
uv run python scripts/ingest.py --input data/videos --out-dir data/voices/my_voice
uv run python scripts/ingest.py --input a.mp4 --input b.m4a --out-dir data/voices/mix --denoise 0.4
```

Then compute latents and clone fast:

```powershell
uv run python scripts/compute_latents.py --speaker-wav data/voices/my_voice/pooled.wav --out data/voices/my_voice/voice.pt
uv run python scripts/clone_fast.py --latents data/voices/my_voice/voice.pt --text "こんにちは！" --out outputs/x.wav --language ja
```

Tip: one speaker per folder. Mixing different speakers blurs the voice.

## Fine-tune XTTS-v2 (local)

Zero-shot borrows the base voice. Fine-tuning bakes YOUR voice into the weights:

```powershell
# 1. Prepare dataset (transcribe + chunk to <=11s, 3+ min ideally)
uv run python scripts/prepare_ft.py --input data/voices/my_voice/clean --out data/ft/myvoice --language ja

# 2. Smoke test (proves it fits your VRAM)
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --smoke

# 3. Full run
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --epochs 50

# Resume from checkpoint
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --resume training/run/<run-dir>/checkpoint_N.pth
```

PC lags while training? Use burst mode — trains N epochs, GPU rests, auto-resumes:

```powershell
uv run python scripts/train_xtts.py --dataset data/ft/myvoice --epochs 20 --burst-epochs 2 --rest-seconds 90
```

## Fine-tune on Google Colab (recommended)

Best quality — free T4 GPU or pay for A100.

```powershell
# Opens the Colab Tools app
colab_prep.bat
```

1. **Prepare tab:** fill in settings → **Run Pipeline** → download `.ipynb` + dataset `.zip`
2. **Upload to Colab:** drag both files into Colab's file panel
3. **Run the notebook** cells top to bottom (GPU required)
4. **Save checkpoints** to Google Drive (Cell 6 in notebook)
5. **Install tab:** upload checkpoint ZIP → **Install Model** → restart VoiceStudio

Training tips for realistic voice:
| Audio duration | Quality |
|---|---|
| < 1 min | Robotic |
| 3–5 min | Recognizable |
| **10–30 min** | **Natural ✅** |
| 1+ hour | Professional |

## Export model

```powershell
uv run python scripts/export_model.py --checkpoint training/run/<run>/checkpoint_50.pth --out models/myvoice
```

Or use the **Install Trained Model** tab in `colab_prep.bat` — it does this automatically.

## Calibrate voice

```powershell
uv run python scripts/calibrate.py \
  --text "こんにちは！" \
  --voice data/voices/my_voice/pooled.wav \
  --model models/myvoice \
  --pitch 2 --temperature 0.85 --language ja \
  --out outputs/cali.wav
```

- `--pitch` semitones (-12..+12)
- `--temperature` 0.6 flat → 0.85 expressive
- `--model` bundle dir (omit = base XTTS)

## Engines (pluggable)

`xtts` is the built-in engine. Add third-party engines via `VoiceEngine` interface:

```powershell
uv run python scripts/engine.py list
uv run python scripts/engine.py add git+https://github.com/user/their-engine.git
uv run python scripts/engine.py test --engine xtts --text "hello" --ref data/voices/my_voice/pooled.wav --language en
```

To publish an engine on GitHub (duck-typed, no VoiceStudio dependency needed):

```python
class MyEngine:
    name = "my-engine"; label = "My Engine"; needs_ref = True
    def synthesize(self, text, out, language="en", ref_audio=None, **kw):
        ...  # write WAV to Path(out), return it
```

```toml
# pyproject.toml
[project.entry-points."voicestudio.engines"]
my-engine = "their_package:MyEngine"
```

## Tips for natural voice

- **Reference audio:** 6–12s, one speaker, no music/SFX. Noisy or mixed-speaker clips sound robotic.
- **Match language:** Japanese speaker + `--language en` gives cross-lingual accent. Use same-language first to check quality.
- **Denoise lightly:** `--denoise 0.3` for expressive speech, `0.6` for noisy recordings. Heavy denoise flattens dynamics.
- **Text punctuation:** Use `? ! … 、` and short sentences. `--speed 0.9–0.95` helps naturalness.
- **Temperature:** 0.6 = flat/safe, 0.75 = default, 0.85 = expressive.
- **More data = better:** 10+ minutes of clean audio gives a much more realistic fine-tuned model.

## Troubleshooting

| Error | Fix |
|---|---|
| Port bind error (Errno 13) | Port reserved by Hyper-V. `start.bat` uses port 9999 to avoid this. |
| `EOFError` on first run | Pipe `y`: `"y" \| uv run python scripts/clone.py ...` |
| `BeamSearchScorer` import error | transformers v5 installed. Run `uv sync` to pin 4.46.3. |
| `pkg_resources` missing | Run `uv sync` — pins `setuptools<81`. |
| `No module named 'cutlet'` | Run `uv sync` — required for Japanese. |
| `matplotlib backend` error in Colab | Use the generated `.ipynb` from `colab_prep.bat` — it sets `MPLBACKEND=agg` automatically. |
| `File is not a zip file` when installing | Re-download from Google Drive: right-click folder → **Download**. |


-- Sample model: https://huggingface.co/minta1234/japanese-mix-xtts