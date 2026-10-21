"""VoiceStudio web backend: JSON API + serves the web/ UI.

Run:  uv run python scripts/api.py        -> http://127.0.0.1:8000
Or double-click start.bat (opens the browser for you).

Two API surfaces:
  /api/*     local app endpoints (UI). No auth; bind to 127.0.0.1.
  /api/v1/*  public-style API (fish.audio-like). Requires "Authorization: Bearer <key>".
             Interactive docs at /docs.
"""
import argparse
import contextlib
import importlib.util
import json
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # project root (engines package)
sys.path.insert(0, str(ROOT / "scripts"))  # hw, train_xtts direct imports

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch
import uvicorn
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, Header,
                     HTTPException, Request, UploadFile)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

VOICE_DIRS = [ROOT / "data" / "audio_clean", ROOT / "data" / "voices"]
MODELS_DIR = ROOT / "models"
KEYS_FILE = ROOT / "data" / "api_keys.json"

# ---------------------------------------------------------------------------
# Model management: one model resident at a time (swaps to fit small VRAM).
# ---------------------------------------------------------------------------
_slot: dict = {"key": "__base__", "model": None}
_latents_cache: dict[tuple, dict] = {}


def evict_model():
    """Unload model from GPU memory to free VRAM for training or other tasks."""
    if _slot["model"] is not None:
        del _slot["model"]
        _slot["model"] = None
        _slot["key"] = None
    _latents_cache.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def get_model(model_dir: str | None = None):
    """Load (and cache) base XTTS or a fine-tuned bundle. Swaps on change."""
    key = str(Path(model_dir).resolve()) if model_dir else "__base__"
    if _slot["key"] == key and _slot["model"] is not None:
        return _slot["model"]

    evict_model()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    from TTS.api import TTS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    if model_dir:
        b = Path(model_dir)
        if not (b / "model.pth").exists():
            raise HTTPException(400, f"Not a model bundle: {model_dir}")
        model = TTS(model_path=str(b), config_path=str(b / "config.json")).to(dev)
    else:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    _slot["key"] = key
    _slot["model"] = model
    label = "base xtts-v2" if not model_dir else Path(model_dir).name
    print(f"[api] model load ({label}) on {dev}: {time.perf_counter() - t0:.1f}s")
    return model


def get_latents(speaker_wav: Path, model) -> dict:
    key = (_slot["key"], str(speaker_wav))
    if key not in _latents_cache:
        dev = next(model.synthesizer.tts_model.parameters()).device
        pt = speaker_wav.with_suffix(".pt")
        # .pt sidecars were computed against the base model only.
        if pt.exists() and _slot["key"] == "__base__":
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


# ---------------------------------------------------------------------------
# Voice helpers
# ---------------------------------------------------------------------------
def list_voices() -> list[dict]:
    voices = []
    for d in VOICE_DIRS:
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            voices.append({"id": str(wav.relative_to(ROOT)).replace("\\", "/"),
                           "label": wav.stem, "source": d.name})
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                for wav in sorted(sub.glob("*.wav")):
                    if wav.name == "pooled.wav":
                        voices.append({"id": str(wav.relative_to(ROOT)).replace("\\", "/"),
                                       "label": f"{sub.name} (pooled)",
                                       "source": d.name})
    return voices


def resolve_voice(voice_id: str) -> Path:
    p = (ROOT / voice_id).resolve()
    if not str(p).startswith(str(ROOT)) or not p.is_file():
        raise HTTPException(400, f"Unknown voice: {voice_id}")
    return p


def _hashtag_tags(names: list[str]) -> list[str]:
    tags: list[str] = []
    for n in names:
        for t in re.findall(r"#([A-Za-z0-9_]+)", n):
            t = t.lower()
            if t not in tags:
                tags.append(t)
    return tags[:8]


def list_library() -> list[dict]:
    """fish.audio-style browsable library: pooled voices, loose clips, models."""
    items: list[dict] = []
    vroot = ROOT / "data" / "voices"
    if vroot.is_dir():
        for sub in sorted(vroot.iterdir()):
            if not sub.is_dir():
                continue
            pooled = sub / "pooled.wav"
            if not pooled.exists():
                continue
            seconds, clips, tags = 0.0, 0, []
            mf = sub / "manifest.json"
            if mf.exists():
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                    clips = len(data)
                    seconds = round(sum(float(x.get("seconds", 0) or 0)
                                        for x in data), 1)
                    tags = _hashtag_tags([Path(str(x.get("source", ""))).name
                                          for x in data])
                except Exception:
                    pass
            model = MODELS_DIR / sub.name
            items.append({
                "id": str(pooled.relative_to(ROOT)).replace("\\", "/"),
                "name": sub.name,
                "kind": "voice",
                "seconds": seconds or None,
                "clips": clips,
                "tags": tags,
                "has_latents": (sub / "voice.pt").exists()
                or pooled.with_suffix(".pt").exists(),
                "model": str(model.relative_to(ROOT)).replace("\\", "/")
                if (model / "model.pth").exists() else None,
            })

    ac = ROOT / "data" / "audio_clean"
    if ac.is_dir():
        for wav in sorted(ac.glob("*.wav")):
            items.append({
                "id": str(wav.relative_to(ROOT)).replace("\\", "/"),
                "name": wav.stem,
                "kind": "clip",
                "seconds": None,
                "clips": 1,
                "tags": [],
                "has_latents": wav.with_suffix(".pt").exists(),
                "model": None,
            })

    voice_names = {i["name"] for i in items}
    if MODELS_DIR.is_dir():
        for m in sorted(MODELS_DIR.iterdir()):
            if (m.is_dir() and (m / "model.pth").exists()
                    and m.name not in voice_names):
                items.append({
                    "id": None,
                    "name": m.name,
                    "kind": "model",
                    "seconds": None,
                    "clips": 0,
                    "tags": [],
                    "has_latents": False,
                    "model": str(m.relative_to(ROOT)).replace("\\", "/"),
                })
    return items


def list_models() -> list[dict]:
    out = []
    if MODELS_DIR.is_dir():
        for m in sorted(MODELS_DIR.iterdir()):
            if m.is_dir() and (m / "model.pth").exists():
                out.append({"id": str(m.relative_to(ROOT)).replace("\\", "/"),
                            "name": m.name})
    return out


# ---------------------------------------------------------------------------
# API keys (public-style API auth)
# ---------------------------------------------------------------------------
def _load_keys() -> list[dict]:
    if KEYS_FILE.exists():
        try:
            return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_keys(keys: list[dict]) -> None:
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")


def _mask(k: str) -> str:
    return f"{k[:6]}…{k[-4:]}" if len(k) > 12 else k


def require_key(authorization: str | None = Header(None)) -> dict:
    keys = {k["key"]: k for k in _load_keys()}
    if not keys:
        raise HTTPException(401, "No API keys configured. Create one in the "
                                 "API tab (or POST /api/keys).")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing 'Authorization: Bearer <key>' header.")
    token = authorization.split(" ", 1)[1].strip()
    if token not in keys:
        raise HTTPException(403, "Invalid API key.")
    return keys[token]


# ---------------------------------------------------------------------------
# Synthesis core (shared by /api/clone and /api/v1/tts)
# ---------------------------------------------------------------------------
def _synthesize(text: str, voice: str | None, language: str, engine: str,
                temperature: float, speed: float,
                model_dir: str | None) -> Path:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Empty text.")
    ref = resolve_voice(voice) if voice else None

    if engine != "xtts":
        from engines.registry import get_engine
        try:
            eng = get_engine(engine)
        except KeyError as e:
            raise HTTPException(400, e.args[0] if e.args
                                else f"Unknown engine '{engine}'.")
        if getattr(eng, "needs_ref", True) and ref is None:
            raise HTTPException(400, f"Engine '{engine}' needs a reference voice.")
        out = Path(tempfile.mktemp(suffix=".wav"))
        try:
            eng.synthesize(text, out, language, ref_audio=ref, speed=speed,
                           temperature=temperature)
        except Exception as e:  # bridge down, bad payload, etc. -> JSON, not 500
            raise HTTPException(502, f"Engine '{engine}' failed: {e}")
        return out

    if ref is None:
        raise HTTPException(400, "XTTS needs a reference voice.")
    model = get_model(model_dir)
    lat = get_latents(ref, model)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            wav = model.synthesizer.tts_model.inference(
                text, language, lat["gpt_cond_latent"], lat["speaker_embedding"],
                temperature=temperature, speed=speed,
                enable_text_splitting=True,
            )["wav"]
    except Exception as e:
        raise HTTPException(502, f"XTTS synthesis failed: {e}")
    out = Path(tempfile.mktemp(suffix=".wav"))
    model.synthesizer.save_wav(wav, str(out))
    print(f"[api] synth {time.perf_counter() - t0:.1f}s -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------
class CloneRequest(BaseModel):
    text: str
    voice: str = ""
    language: str = "ja"
    engine: str = "xtts"
    temperature: float = 0.75
    speed: float = 0.95
    model: str | None = None  # fine-tuned bundle dir, e.g. "models/myvoice"


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    model: str | None = None
    language: str = "ja"
    engine: str = "xtts"
    temperature: float = 0.75
    speed: float = 0.95


class KeyCreate(BaseModel):
    name: str = "default"


class MoodRequest(BaseModel):
    text: str
    mode: str = "auto"
    language: str = "auto"


class DialogueMoodRequest(BaseModel):
    script: str
    mode: str = "auto"
    engine: str = "fish-speech"


app = FastAPI(
    title="VoiceStudio API",
    version="2.0",
    description="Local voice-cloning TTS platform. The public-style API lives "
                "under `/api/v1` and requires a Bearer key (see the API tab).",
)


@app.middleware("http")
async def no_cache_ui(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path in ("/", "/index.html") or request.url.path.endswith(".html"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@app.get("/api/health", tags=["app"])
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "model": _slot["key"] if _slot["model"] is not None else None}


@app.get("/api/voices", tags=["app"])
def voices():
    return {"voices": list_voices()}


@app.get("/api/library", tags=["app"])
def library():
    return {"items": list_library()}


@app.get("/api/models", tags=["app"])
def models():
    return {"models": list_models()}


@app.get("/api/preview", tags=["app"])
def preview(id: str):
    return FileResponse(str(resolve_voice(id)), media_type="audio/wav")


@app.get("/api/engines", tags=["app"])
def engines():
    from engines.registry import list_engines
    out = []
    for n, c in sorted(list_engines().items()):
        out.append({
            "id": n,
            "label": getattr(c, "label", n),
            "needs_ref": getattr(c, "needs_ref", True),
            "supports_direct": getattr(c, "supports_direct", False),
            "supports_zero_shot": getattr(c, "supports_zero_shot", False),
        })
    return {"engines": out}


@app.post("/api/mood/analyze", tags=["mood"])
def analyze_mood_endpoint(req: MoodRequest):
    """Analyzes text mood, emotion scores, and TTS prosody."""
    from mood import MoodAnalyzer
    analyzer = MoodAnalyzer()
    res = analyzer.analyze(req.text, mode=req.mode, language=req.language)
    return res.to_dict()


@app.post("/api/mood/dialogue", tags=["mood"])
def analyze_dialogue_mood_endpoint(req: DialogueMoodRequest):
    """Analyzes a multi-line or multi-character dialogue script."""
    from mood import MoodAnalyzer
    analyzer = MoodAnalyzer()
    results = analyzer.analyze_dialogue(req.script, mode=req.mode)
    tagged_script = analyzer.format_tts_script(req.script, engine=req.engine)
    return {
        "lines": [r.to_dict() for r in results],
        "tagged_script": tagged_script,
    }


def _script_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ingest_module():
    return _script_module("audio_ingest")


@app.get("/api/hw", tags=["app"])
def hw():
    mod = _script_module("hw_detect")
    info = mod.detect()
    rust_bin = ROOT / "target" / "release" / "voice-prep.exe"
    if not rust_bin.exists():
        rust_bin = ROOT / "crates" / "voice_core" / "target" / "release" / "voice-prep.exe"
    return {
        **info,
        "profile": mod.recommend(info),
        "profiles": list(mod.PROFILES),
        "rust_accelerated": rust_bin.exists(),
        "rust_version": "0.1.0 (Burn + Sinc DSP)",
    }



@app.get("/api/ft-sources", tags=["app"])
def ft_sources():
    srcs = []
    ac = ROOT / "data" / "audio_clean"
    if ac.is_dir() and list(ac.glob("*.wav")):
        srcs.append({"id": "data/audio_clean", "label": "audio_clean"})
    vroot = ROOT / "data" / "voices"
    if vroot.is_dir():
        for sub in sorted(vroot.iterdir()):
            if sub.is_dir() and list((sub / "clean").glob("*.wav")):
                srcs.append({"id": str((sub / 'clean').relative_to(ROOT)).replace("\\", "/"),
                             "label": f"{sub.name}/clean"})
    return {"sources": srcs}


class TrainRequest(BaseModel):
    source: str
    language: str = "ja"
    epochs: int = 3
    profile: str = "auto"
    smoke: bool = False
    burst_epochs: int = 0
    rest_seconds: int = 90


_jobs: dict[str, dict] = {}


def _run_train_job(job_id: str, req: TrainRequest):
    job = _jobs[job_id]
    log = open(job["log"], "w", encoding="utf-8")
    try:
        # Crucial for low VRAM (4GB RTX 3050): Evict inference model from VRAM before training!
        evict_model()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            ds = ROOT / "training" / "web" / job_id
            rust_bin = ROOT / "target" / "release" / "voice-prep.exe"
            if not rust_bin.exists():
                rust_bin = ROOT / "crates" / "voice_core" / "target" / "release" / "voice-prep.exe"

            if rust_bin.exists():
                print(f"[api.train] ⚡ Using compiled Rust voice-prep engine: {rust_bin}")
                import subprocess
                cmd = [
                    str(rust_bin), "batch-prep",
                    "--input-dir", str((ROOT / req.source).resolve()),
                    "--out-dir", str(ds.resolve()),
                    "--sr", "22050",
                    "--max-dur", "11.0",
                ]
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print("[api.train] Rust batch-prep returned error, fallback to Python train_prepare...")
                    prep = _script_module("train_prepare")
                    prep.prepare(ROOT / req.source, ds, req.language, job_id)
            else:
                prep = _script_module("train_prepare")
                prep.prepare(ROOT / req.source, ds, req.language, job_id)

            tr = _script_module("train_xtts")
            tr.train(ds, req.language, ds / "run", req.epochs, False,
                     req.smoke, None, str(tr.BASE_DIR),
                     "こんにちは、これはテストです。",
                     None, None, False, None, 50, "auto"
                     if req.profile == "auto" else req.profile,
                     f"web-{job_id}", req.burst_epochs, req.rest_seconds)
        job["status"] = "done"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[job {job_id}] FAILED: {e}\n{tb}", file=log)
        job["status"] = "error"
    finally:
        log.close()


@app.post("/api/train", tags=["app"])
def start_train(req: TrainRequest):
    src = (ROOT / req.source).resolve()
    if not str(src).startswith(str(ROOT)) or not src.is_dir():
        raise HTTPException(400, f"Unknown source: {req.source}")
    if any(j["status"] == "running" for j in _jobs.values()):
        raise HTTPException(409, "A training job is already running.")
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "source": req.source,
                     "log": str(ROOT / "training" / "web" / f"{job_id}.log"),
                     "started": time.strftime("%H:%M:%S")}
    Path(_jobs[job_id]["log"]).parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_run_train_job, args=(job_id, req),
                     daemon=True).start()
    return {"job": job_id}


@app.get("/api/train", tags=["app"])
def list_train():
    return {"jobs": [{**{k: v for k, v in j.items() if k != "log"}, "id": i}
                     for i, j in _jobs.items()]}


@app.get("/api/train/{job_id}", tags=["app"])
def train_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job.")
    log = Path(_jobs[job_id]["log"])
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:] \
        if log.exists() else []
    return {"id": job_id, "status": _jobs[job_id]["status"], "log": tail}


@app.post("/api/voices", tags=["app"])
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
    voice_id = str(pooled.relative_to(ROOT)).replace("\\", "/")
    return {"voice": voice_id, "label": f"{safe} (pooled)",
            "files": len(saved)}


@app.post("/api/clone", tags=["app"])
def clone(req: CloneRequest, bg: BackgroundTasks):
    out = _synthesize(req.text, req.voice or None, req.language, req.engine,
                      req.temperature, req.speed, req.model)
    latest_wav = ROOT / "outputs" / "latest.wav"
    try:
        latest_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, latest_wav)
    except Exception as e:
        print(f"[api] could not copy to latest.wav: {e}")

    bg.add_task(out.unlink, missing_ok=True)
    return FileResponse(str(latest_wav), media_type="audio/wav",
                        filename="voicestudio_cloned.wav")


@app.get("/api/download/latest", tags=["app"])
def download_latest():
    latest_wav = ROOT / "outputs" / "latest.wav"
    if not latest_wav.exists():
        raise HTTPException(404, "No audio generated yet.")
    stamp = int(time.time())
    fname = f"voice_{stamp}.wav"
    return FileResponse(
        str(latest_wav),
        media_type="audio/wav",
        filename=fname,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )



# ---- API key management (local admin) ----
@app.get("/api/keys", tags=["keys"])
def list_keys():
    return {"keys": [{"id": k["key"][:8], "name": k.get("name", "default"),
                      "created": k.get("created", ""), "preview": _mask(k["key"])}
                     for k in _load_keys()]}


@app.post("/api/keys", tags=["keys"])
def create_key(req: KeyCreate):
    keys = _load_keys()
    key = "vs_" + secrets.token_urlsafe(24)
    keys.append({"key": key, "name": (req.name or "default").strip(),
                 "created": time.strftime("%Y-%m-%d %H:%M")})
    _save_keys(keys)
    return {"key": key, "name": req.name}  # full key shown once


@app.delete("/api/keys/{key_id}", tags=["keys"])
def revoke_key(key_id: str):
    keys = _load_keys()
    new = [k for k in keys if k["key"][:8] != key_id]
    if len(new) == len(keys):
        raise HTTPException(404, "Unknown key.")
    _save_keys(new)
    return {"ok": True, "remaining": len(new)}


# ---- Public-style API (Bearer key required) ----
@app.post("/api/v1/tts", tags=["v1"], dependencies=[Depends(require_key)])
def v1_tts(req: TTSRequest, bg: BackgroundTasks):
    """Generate speech. Authenticate with `Authorization: Bearer <key>`."""
    out = _synthesize(req.text, req.voice, req.language, req.engine,
                      req.temperature, req.speed, req.model)
    bg.add_task(out.unlink, missing_ok=True)
    return FileResponse(str(out), media_type="audio/wav",
                        filename="tts.wav", background=bg)


@app.get("/api/v1/library", tags=["v1"], dependencies=[Depends(require_key)])
def v1_library():
    return {"items": list_library()}


app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    get_model()  # warm up before serving
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
