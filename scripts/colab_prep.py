"""Colab Training Prep — Gradio App

Step-by-step pipeline:
  1. Ingest  : extract + clean audio from video files
  2. Prepare : transcribe + segment for XTTS fine-tune
  3. Pack    : zip dataset + generate .ipynb ready for Colab
  4. Upload  : push zip + notebook to Google Drive (accessible from Colab)

Run:
  uv run python scripts/colab_prep.py
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
VOICES_DIR = DATA_DIR / "voices"
FT_DIR = DATA_DIR / "ft"
NOTEBOOKS_DIR = ROOT


# ---------------------------------------------------------------------------
# Notebook generator
# ---------------------------------------------------------------------------

def generate_notebook(dataset_name: str, language: str, epochs: int) -> Path:
    nb_path = ROOT / f"colab_train_{dataset_name}.ipynb"
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# VoiceStudio: Train XTTS-v2 — `{dataset_name}`\n",
                    "\n",
                    f"Language: **{language}** | Epochs: **{epochs}**\n",
                    "\n",
                    "> **Before you start:** Go to **Runtime > Change runtime type** → **T4 GPU**\n",
                    "\n",
                    "### Files to upload before running:\n",
                    "1. `VoiceStudio.zip` — project code\n",
                    f"2. `{dataset_name}_dataset.zip` — prepared dataset\n",
                    "\n",
                    "Drag both files into the Colab left-panel file explorer."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 0. Check GPU\n", "!nvidia-smi"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Mount Google Drive (saves checkpoints safely)\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Install uv + unzip project + install dependencies\n",
                    "!curl -LsSf https://astral.sh/uv/install.sh | sh\n",
                    "import os\n",
                    "os.environ['PATH'] = f\"/root/.cargo/bin:{os.environ['PATH']}\"\n",
                    "\n",
                    "!unzip -q VoiceStudio.zip\n",
                    "%cd VoiceStudio\n",
                    "!uv sync"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Unzip dataset\n",
                    f"!mkdir -p data/ft/{dataset_name}\n",
                    f"!unzip -q /content/{dataset_name}_dataset.zip -d data/ft/{dataset_name}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Download XTTS-v2 base model (~1.9 GB)\n",
                    "!env MPLBACKEND=agg uv run python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Train!\n",
                    f"!env MPLBACKEND=agg uv run python scripts/train_xtts.py \\\n",
                    f"    --dataset data/ft/{dataset_name} \\\n",
                    f"    --language {language} \\\n",
                    f"    --epochs {epochs} \\\n",
                    "    --base-dir /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 6. Save checkpoints to Google Drive\n",
                    f"!cp -r training/run /content/drive/MyDrive/VoiceStudio_{dataset_name}\n",
                    "print('Saved to Google Drive! Download from drive.google.com')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    return nb_path


# ---------------------------------------------------------------------------
# Pipeline steps (streaming subprocess output)
# ---------------------------------------------------------------------------

def _run_stream(cmd: list[str], cwd: Path):
    """Yield output lines from a subprocess."""
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    if proc.returncode != 0:
        yield f"\n❌ Command failed with exit code {proc.returncode}\n"
    else:
        yield f"\n✅ Done!\n"


def run_pipeline(voice_name: str, language: str, epochs: int, video_folder: str, denoise: float):
    if not voice_name.strip():
        yield "⚠️ Please set a model name first.", "", gr.update(visible=False), gr.update(visible=False)
        return

    uv = shutil.which("uv") or "uv"
    voice_dir = VOICES_DIR / voice_name
    ft_dir = FT_DIR / voice_name
    video_path = Path(video_folder) if video_folder else VIDEOS_DIR
    log = ""

    # ---- Step 1: Ingest ----
    log += "=" * 50 + "\n📥 Step 1: Ingest videos\n" + "=" * 50 + "\n"
    yield log, "🔄 Step 1/3: Ingesting...", gr.update(visible=False), gr.update(visible=False)

    cmd = [uv, "run", "python", "scripts/ingest.py",
           "--input", str(video_path),
           "--out-dir", str(voice_dir),
           "--denoise", str(denoise)]
    for line in _run_stream(cmd, ROOT):
        log += line
        yield log, "🔄 Step 1/3: Ingesting...", gr.update(visible=False), gr.update(visible=False)

    # ---- Step 2: Prepare ----
    log += "\n" + "=" * 50 + "\n🎙️ Step 2: Prepare dataset\n" + "=" * 50 + "\n"
    yield log, "🔄 Step 2/3: Preparing...", gr.update(visible=False), gr.update(visible=False)

    clean_dir = voice_dir / "clean"
    cmd = [uv, "run", "python", "scripts/prepare_ft.py",
           "--input", str(clean_dir),
           "--out", str(ft_dir),
           "--language", language]
    for line in _run_stream(cmd, ROOT):
        log += line
        yield log, "🔄 Step 2/3: Preparing...", gr.update(visible=False), gr.update(visible=False)

    # ---- Step 3: Pack ----
    log += "\n" + "=" * 50 + "\n📦 Step 3: Pack ZIP + Notebook\n" + "=" * 50 + "\n"
    yield log, "🔄 Step 3/3: Packing...", gr.update(visible=False), gr.update(visible=False)

    # Create dataset zip
    zip_path = FT_DIR / f"{voice_name}_dataset.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in ft_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(ft_dir))
        log += f"  ✅ Dataset zip: {zip_path}\n"
    except Exception as e:
        log += f"  ❌ Zip failed: {e}\n"

    # Create project zip (only important files)
    proj_zip = ROOT / "VoiceStudio.zip"
    INCLUDE_DIRS = {"scripts", "engines"}
    INCLUDE_FILES = {"pyproject.toml", "uv.lock", "README.md"}
    
    try:
        with zipfile.ZipFile(proj_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in ROOT.rglob("*"):
                if not f.is_file():
                    continue
                
                # Exclude __pycache__
                if "__pycache__" in f.parts:
                    continue
                
                rel_path = f.relative_to(ROOT)
                top_level = rel_path.parts[0]
                
                if top_level in INCLUDE_DIRS or str(rel_path) in INCLUDE_FILES:
                    zf.write(f, rel_path)
                    
        log += f"  ✅ Project zip: {proj_zip}\n"
    except Exception as e:
        log += f"  ❌ Project zip failed: {e}\n"

    # Generate notebook
    nb_path = generate_notebook(voice_name, language, epochs)
    log += f"  ✅ Notebook: {nb_path}\n"

    log += "\n🎉 All done! Files ready:\n"
    log += f"  📁 {zip_path}\n"
    log += f"  📁 {proj_zip}\n"
    log += f"  📓 {nb_path}\n"
    log += "\nUpload these to Google Colab → Run the notebook!\n"

    yield (log, "✅ Complete! Files ready to upload.",
           gr.update(visible=True, value=str(nb_path)),
           gr.update(visible=True, value=str(zip_path)))


# ---------------------------------------------------------------------------
# Upload to Google Drive
# ---------------------------------------------------------------------------

def upload_to_drive(nb_path: str, zip_path: str):
    """Upload notebook + dataset zip to Google Drive root."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        return "❌ Google API libraries not found. Run: uv add google-api-python-client google-auth-oauthlib"

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds_file = ROOT / ".gdrive_token.json"
    client_secrets = ROOT / "client_secrets.json"

    if not client_secrets.exists():
        return (
            "❌ Missing `client_secrets.json` in the project folder.\n\n"
            "How to create one:\n"
            "1. Go to https://console.cloud.google.com/\n"
            "2. Create a new project\n"
            "3. Enable Google Drive API\n"
            "4. Create credentials → OAuth 2.0 → Desktop app\n"
            "5. Download the JSON file and save it in the project folder as `client_secrets.json`"
        )

    creds = None
    if creds_file.exists():
        creds = Credentials.from_authorized_user_file(str(creds_file), SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)
        creds_file.write_text(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    log = ""

    for fpath in [nb_path, zip_path]:
        if not fpath or not Path(fpath).exists():
            continue
        fpath = Path(fpath)
        mime = "application/zip" if fpath.suffix == ".zip" else "application/octet-stream"
        media = MediaFileUpload(str(fpath), mimetype=mime, resumable=True)
        meta = {"name": fpath.name}
        result = service.files().create(body=meta, media_body=media, fields="id,name").execute()
        log += f"✅ Uploaded: {result['name']} (id={result['id']})\n"

    log += "\n📁 Go to drive.google.com to view your uploaded files!"
    return log


# ---------------------------------------------------------------------------
# Import Trained Model
# ---------------------------------------------------------------------------

def import_model(zip_file: str, model_name: str):
    if not zip_file:
        return "❌ Please upload a ZIP file."
    if not model_name.strip():
        return "❌ Please enter a model name."

    log = f"🔄 Processing {Path(zip_file).name}...\n"
    out_dir = ROOT / "models" / model_name.strip()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(tmp_path)
            log += "✅ Unzipped successfully.\n"
        except Exception as e:
            return log + f"❌ Failed to unzip: {e}"

        # Find checkpoint
        chkpts = list(tmp_path.rglob("checkpoint_*.pth"))
        if not chkpts:
            # Maybe it's already exported? Look for model.pth
            if list(tmp_path.rglob("model.pth")):
                shutil.copytree(tmp_path, out_dir, dirs_exist_ok=True)
                return log + f"✅ Installed directly to models/{model_name}"
            return log + "❌ No checkpoint_*.pth or model.pth found in ZIP."
            
        # Get the latest checkpoint by number
        def get_step(p):
            import re
            m = re.search(r"checkpoint_(\d+)\.pth", p.name)
            return int(m.group(1)) if m else 0
            
        best_chkpt = max(chkpts, key=get_step)
        log += f"✅ Found checkpoint: {best_chkpt.name}\n"
        
        # Export model
        log += f"⏳ Exporting to models/{model_name} (this might take a minute)...\n"
        cmd = [shutil.which("uv") or "uv", "run", "python", "scripts/export_model.py", 
               "--checkpoint", str(best_chkpt), "--out", str(out_dir)]
        
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode == 0:
            log += f"🎉 Success! Model installed to models/{model_name}\n"
            log += "👉 You can now restart VoiceStudio (start.bat) to use it!"
        else:
            log += f"❌ Export failed:\n{proc.stderr}"
            
    return log


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="VoiceStudio — Colab Prep",
) as app:

    gr.Markdown("# 🎙️ VoiceStudio — Colab Tools")
    
    with gr.Tabs():
        with gr.TabItem("1. Prepare for Colab"):
            gr.Markdown("**Pipeline:** Ingest videos → Prepare dataset → Pack ZIP + Notebook → Upload to Google Drive")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    voice_name_in = gr.Textbox(label="Voice Name", placeholder="my_voice", value="japanese_mix")
                    language_in = gr.Dropdown(
                        label="Language", value="ja",
                        choices=[("Japanese 🇯🇵", "ja"), ("English 🇬🇧", "en"),
                                 ("Thai 🇹🇭", "th"), ("Chinese 🇨🇳", "zh")]
                    )
                    epochs_in = gr.Slider(label="Epochs", minimum=10, maximum=200, value=50, step=10)
                    video_folder_in = gr.Textbox(
                        label="Video Folder",
                        placeholder="data/videos (default)",
                        value=str(VIDEOS_DIR)
                    )
                    denoise_in = gr.Slider(label="Denoise strength", minimum=0.1, maximum=0.9, value=0.3, step=0.1)

                    run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Progress")
                    status_out = gr.Markdown("⏳ Ready. Press **Run Pipeline** to start.", elem_classes="status-bar")
                    log_out = gr.Code(label="Live Log", language=None, lines=20)

            gr.Markdown("---")
            gr.Markdown("### 📁 Output Files")

            with gr.Row():
                nb_file = gr.File(label="📓 Notebook (.ipynb)", visible=False)
                zip_file = gr.File(label="📦 Dataset ZIP", visible=False)

            gr.Markdown("---")
            gr.Markdown("### ☁️ Upload to Google Drive")
            gr.Markdown(
                "> You must have `client_secrets.json` in the project folder first."
            )
            upload_btn = gr.Button("⬆️ Upload to Google Drive", variant="secondary")
            upload_log = gr.Textbox(label="Upload Log", lines=5)
            
        with gr.TabItem("2. Install Trained Model"):
            gr.Markdown("Upload the trained model ZIP file downloaded from Colab (Google Drive) to automatically install it into VoiceStudio!")
            with gr.Row():
                with gr.Column():
                    model_zip_in = gr.File(label="📦 Upload Trained Model ZIP", file_types=[".zip"])
                    model_name_in = gr.Textbox(label="Model Name (e.g., my_japanese_voice)", value="my_new_voice")
                    install_btn = gr.Button("📥 Install Model", variant="primary")
                with gr.Column():
                    install_log_out = gr.Textbox(label="Install Log", lines=10)

    # Events
    run_btn.click(
        fn=run_pipeline,
        inputs=[voice_name_in, language_in, epochs_in, video_folder_in, denoise_in],
        outputs=[log_out, status_out, nb_file, zip_file],
        show_progress=False
    )

    upload_btn.click(
        fn=upload_to_drive,
        inputs=[nb_file, zip_file],
        outputs=[upload_log]
    )
    
    install_btn.click(
        fn=import_model,
        inputs=[model_zip_in, model_name_in],
        outputs=[install_log_out]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=9998, inbrowser=True)
