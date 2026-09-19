"""VoiceStudio web backend: JSON API + serves the web/ UI.

Run:  uv run python scripts/api.py        -> http://127.0.0.1:8000
Or double-click start.bat (opens the browser for you).
"""
import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # project root (engines package)

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

VOICE_DIRS = [ROOT / "data" / "audio_clean", ROOT / "data" / "voices"]

_model = None
_latents_cache: dict[str, dict] = {}


def get_model():
    global _model
    if _model is None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        from TTS.api import TTS

        t0 = time.perf_counter()
        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[api] model load: {time.perf_counter() - t0:.1f}s")
    return _model


def list_voices() -> list[dict]:
    voices = []
    for d in VOICE_DIRS:
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            voices.append({"id": str(wav.relative_to(ROOT)), "label": wav.stem,
                           "source": d.name})
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                for wav in sorted(sub.glob("*.wav")):
                    if wav.name == "pooled.wav":
                        voices.append({"id": str(wav.relative_to(ROOT)),
                                       "label": f"{sub.name} (pooled)",
                                       "source": d.name})
    return voices


def resolve_voice(voice_id: str) -> Path:
    p = (ROOT / voice_id).resolve()
    if not str(p).startswith(str(ROOT)) or not p.is_file():
        raise HTTPException(400, f"Unknown voice: {voice_id}")
    return p


def get_latents(speaker_wav: Path) -> dict:
    key = str(speaker_wav)
    if key not in _latents_cache:
        model = get_model()
        dev = next(model.synthesizer.tts_model.parameters()).device
        pt = speaker_wav.with_suffix(".pt")
        if pt.exists():
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
        else:
            gpt, spk = model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=str(speaker_wav))
            d = {"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}
        _latents_cache[key] = {
            "gpt_cond_latent": d["gpt_cond_latent"].to(dev),
            "speaker_embedding": d["speaker_embedding"].to(dev),
        }
    return _latents_cache[key]


class CloneRequest(BaseModel):
    text: str
    voice: str
    language: str = "ja"
    engine: str = "xtts"
    temperature: float = 0.75
    speed: float = 0.95


app = FastAPI(title="VoiceStudio")


@app.get("/api/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available()}


@app.get("/api/voices")
def voices():
    return {"voices": list_voices()}


@app.get("/api/engines")
def engines():
    from engines.registry import list_engines

    return {"engines": [{"id": n, "label": c().label}
                        for n, c in sorted(list_engines().items())]}


def _ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest", ROOT / "scripts" / "ingest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.post("/api/voices")
async def create_voice(name: str = Form("my_voice"),
                       denoise: float = Form(0.5),
                       files: list[UploadFile] = File(...)):
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "voice"
    vdir = ROOT / "data" / "voices" / safe
    updir = vdir / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        fname = Path(f.filename or "upload.bin").name
        dest = updir / fname
        with open(dest, "wb") as o:
            shutil.copyfileobj(f.file, o)
        saved.append(str(dest))
    if not saved:
        raise HTTPException(400, "No files received.")
    pooled = _ingest_module().ingest(saved, vdir, denoise=denoise)
    if pooled is None:
        raise HTTPException(422, "All files failed — see server log.")
    voice_id = str(pooled.relative_to(ROOT))
    return {"voice": voice_id, "label": f"{safe} (pooled)",
            "files": len(saved)}


@app.post("/api/clone")
def clone(req: CloneRequest, bg: BackgroundTasks):
    if not req.text.strip():
        raise HTTPException(400, "Empty text.")
    ref = resolve_voice(req.voice)
    if req.engine != "xtts":
        from engines.registry import get_engine

        out = Path(tempfile.mktemp(suffix=".wav"))
        get_engine(req.engine).synthesize(req.text.strip(), out, req.language,
                                          ref_audio=ref, speed=req.speed)
        bg.add_task(out.unlink, missing_ok=True)
        return FileResponse(str(out), media_type="audio/wav",
                            filename="cloned.wav", background=bg)
    model = get_model()
    lat = get_latents(ref)
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = model.synthesizer.tts_model.inference(
            req.text.strip(), req.language,
            lat["gpt_cond_latent"], lat["speaker_embedding"],
            temperature=req.temperature, speed=req.speed,
            enable_text_splitting=True,
        )["wav"]
    out = Path(tempfile.mktemp(suffix=".wav"))
    model.synthesizer.save_wav(wav, str(out))
    print(f"[api] clone {time.perf_counter() - t0:.1f}s -> {out.name}")
    bg.add_task(out.unlink, missing_ok=True)
    return FileResponse(str(out), media_type="audio/wav", filename="cloned.wav",
                        background=bg)


app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    get_model()  # warm up before serving
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
