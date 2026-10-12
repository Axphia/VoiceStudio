# VoiceStudio

Zero-shot and fine-tuned voice cloning from video files with Coqui XTTS-v2. Extract audio from any video, clean it, and clone — or fine-tune your own model on Google Colab.

Tested on Windows + Python 3.10 + RTX 3050 Laptop (CUDA 12.1).

## Quick Start (no commands)

Double-click **`start.bat`** — the server starts and your browser opens at http://127.0.0.1:9999.
The app has four views in the left sidebar:

- **Library** — browse every cloned voice, reference clip and trained model as cards (tags, duration, preview ▶). Click *Use in Studio* to load one into the playground.
- **Studio** — pick a reference voice + model + engine, type text, press **Generate speech**, and play the result on the waveform player. Build a new voice from uploaded files here too.
- **Train** — fine-tune XTTS as a background job with a live log.
- **API** — create/revoke API keys and read the HTTP API reference (interactive docs at `/docs`).

> **Note:** Port is **9999** (not 8000) to avoid Windows Hyper-V reserved port conflicts.

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
  audio_ingest.py   # batch: many files/folder → cleaned dataset + pooled voice
  audio_clean_ref.py# trim + denoise + normalize ref (e.g. 6s clean audio)
  audio_extract.py  # video → mono WAV
  audio_latents.py  # speaker conditioning → cached .pt (run once per voice)
  clone_voice.py    # XTTS-v2 zero-shot cloning (single shot)
  clone_voice_fast.py # model loads once + cached latents, batch --texts file
  clone_voice_6sec.py # instant 6-second clean voice cloning
  train_prepare.py  # transcribe + segment for XTTS fine-tuning
  train_xtts.py     # XTTS-v2 fine-tuning (local or Colab)
  train_export.py   # checkpoint → portable model bundle
  colab_prep.py     # Gradio app: pipeline prep + model install
  hw_detect.py      # hardware profile detection
  hw_calibrate.py   # tune pitch / temperature on exported model
  engine_manage.py  # engine plugin management CLI
  mood_analyze.py   # text emotion, mood & prosody analysis CLI
  test_mood_sound.py# voice synthesis testing with emotional prosody
web/                # vanilla HTML/JS UI (no build step)
start.bat           # launch VoiceStudio → http://127.0.0.1:9999
colab_prep.bat      # launch Colab Tools → http://127.0.0.1:9998
colab_train_*.ipynb # auto-generated Colab training notebooks
```

## Usage (command line)

1. Extract audio from video:

```powershell
uv run python scripts/audio_extract.py --video "data/videos/YOUR_VIDEO.mp4" --out "data/audio_clean/ref.wav"
```

2. Clean reference (6–10s clean single-speaker is ideal):

```powershell
uv run python scripts/audio_clean_ref.py --in "data/audio_clean/ref.wav" --out "data/audio_clean/clean.wav" --start 3.0 --dur 6.0
```

3. Clone:

```powershell
uv run python scripts/clone_voice.py --text "Hello world" --speaker-wav "data/audio_clean/clean.wav" --out "outputs/cloned.wav" --language en
```

Options: `--language en|ja|zh|...`, `--speed 0.95` (slower, more natural), `--no-split` (disable sentence splitting).

## Batch ingest from many files

Point at a folder or several files — videos and audios mix freely:

```powershell
uv run python scripts/audio_ingest.py --input data/videos --out-dir data/voices/my_voice
uv run python scripts/audio_ingest.py --input a.mp4 --input b.m4a --out-dir data/voices/mix --denoise 0.4
```

Then compute latents and clone fast:

```powershell
uv run python scripts/audio_latents.py --speaker-wav data/voices/my_voice/pooled.wav --out data/voices/my_voice/voice.pt
uv run python scripts/clone_voice_fast.py --latents data/voices/my_voice/voice.pt --text "こんにちは！" --out outputs/x.wav --language ja
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

Three engines ship built-in and appear in the Studio picker:

| Engine | Type | Voice source | Needs external server |
|---|---|---|---|
| `xtts` | Coqui XTTS-v2 | zero-shot clone (reference wav) | no (local, default) |
| `fish-speech` | Fish Audio S1/S2 | zero-shot clone or stored `reference_id` | yes (its `api_server`) |
| `bert-vits2` | BERT + VITS2 | trained speaker (`sid`) + emotion ref | yes (adapter server) |

Each reports capabilities (`needs_ref`, `supports_zero_shot`, `supports_direct`)
via `GET /api/engines`, shown as badges in Studio. Add third-party engines via the `VoiceEngine` interface:

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

## Fish Speech engine (HTTP bridge) — the fish.audio model

`fish-speech` bridges to [Fish Audio S1/S2](https://github.com/fishaudio/fish-speech),
the Dual-AR LLM TTS behind fish.audio: **zero-shot** cloning from a short
reference, 80+ languages, and inline emotion/prosody tags typed into the text
(`[whisper]`, `[excited]`, `[angry]`, …). It ships its own API server, so this is
a thin client — no adapter to write.

```powershell
# 1. In a SEPARATE venv, install fish-speech and download a model checkpoint.
# 2. Start its official API server (default port 8080):
python -m tools.api_server --listen 127.0.0.1:8080

# 3. Point VoiceStudio at it, then pick "Fish Speech S1/S2" in Studio:
set FISH_SPEECH_URL=http://127.0.0.1:8080/v1/tts
set FISH_SPEECH_TOKEN=...          # optional, if the server uses an api key
set FISH_SPEECH_REF_TEXT="..."     # optional transcript of the reference (better cloning)
set FISH_SPEECH_REF_ID=...         # optional server-stored voice id (fish.audio/m/<id>)
start.bat
```

The Studio **reference voice** is sent as `references[0]` (base64 audio) for
in-context cloning; **Expressiveness** maps to `temperature`. Contract:
`POST /v1/tts {text, references[], reference_id, format, temperature, top_p,
repetition_penalty, chunk_length, normalize, streaming}` → `audio/wav`.

## Bert-VITS2 engine (HTTP bridge)

`bert-vits2` is a second engine, selectable in Studio. Bert-VITS2 speaks with a
**trained speaker** (`sid`), not zero-shot cloning, and its deps conflict with
XTTS — so VoiceStudio does **not** import it. Instead `engines/bert_vits2.py` is
a thin HTTP client, and `engines/bert_vits2_server.py` is a reference adapter you
run inside Bert-VITS2's own venv.

```powershell
# 1. In a SEPARATE venv, clone Bert-VITS2 + install its requirements.txt,
#    and download a model (G_*.pth), config.json, bert/ and emotional/ weights.

# 2. From the Bert-VITS2 repo root, start the adapter:
set BV_REPO=C:\path\to\Bert-VITS2
set BV_MODEL=C:\path\to\G_xxx.pth
set BV_CONFIG=C:\path\to\config.json
python C:\path\to\VoiceStudio\engines\bert_vits2_server.py     # serves :5000

# 3. Point VoiceStudio at it, then pick "Bert-VITS2 (HTTP bridge)" in Studio:
set BERT_VITS2_URL=http://127.0.0.1:5000/tts
set BERT_VITS2_SPEAKER=YourSpeaker        # optional; else adapter's first
start.bat
```

Contract: `POST /tts {text, speaker, language, sdp_ratio, noise_scale,
noise_scale_w, length_scale, emotion, reference_audio_b64?}` → `audio/wav`.
The reference voice you pick in Studio is sent as `reference_audio_b64` and used
for **emotion** (CLAP), not timbre. `GET /speakers` lists trained speakers.

## HTTP API (fish.audio-style)

Two surfaces share one server:

- `/api/*` — local app endpoints used by the web UI (no auth; bind `127.0.0.1`).
- `/api/v1/*` — public-style API guarded by a Bearer key. Create keys in the **API** tab (stored in `data/api_keys.json`, git-ignored). Interactive docs: `/docs`.

```bash
# 1. Create a key (local admin endpoint)
curl -X POST http://127.0.0.1:9999/api/keys -H "Content-Type: application/json" -d '{"name":"my-app"}'
# -> {"key":"vs_...","name":"my-app"}   (shown once)

# 2. Generate speech
curl -X POST http://127.0.0.1:9999/api/v1/tts \
  -H "Authorization: Bearer vs_..." \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは！","voice":"data/voices/japanese_mix/pooled.wav","model":"models/japanese_mix","language":"ja"}' \
  --output out.wav

# 3. List available voices
curl http://127.0.0.1:9999/api/v1/library -H "Authorization: Bearer vs_..."
```

`POST /api/v1/tts` body: `text` (required), `voice` (library id, e.g. `data/voices/<name>/pooled.wav`), `model` (optional fine-tuned bundle `models/<name>`; omit = base XTTS-v2), `language`, `engine`, `temperature`, `speed`. Returns `audio/wav`.

Useful local endpoints: `GET /api/library` (voices + clips + models with tags/duration), `GET /api/engines` (capabilities: `needs_ref`, `supports_zero_shot`, `supports_direct`), `GET /api/models`, `GET /api/preview?id=<voice>`.

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