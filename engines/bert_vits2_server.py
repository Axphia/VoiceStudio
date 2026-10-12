"""Reference Bert-VITS2 adapter server — run this INSIDE your Bert-VITS2 venv.

VoiceStudio's ``engines/bert_vits2.py`` is a thin HTTP bridge; it does not import
Bert-VITS2 (whose deps conflict with XTTS). This adapter is the other half of the
bridge: it loads a Bert-VITS2 model once and serves the simple REST contract the
engine calls.

    POST /tts   {text, speaker, language, sdp_ratio, noise_scale, noise_scale_w,
                 length_scale, emotion, reference_audio_b64?}   -> audio/wav
    GET  /speakers                                             -> {"speakers":[...]}

Setup
-----
1. Clone Bert-VITS2 and install ITS ``requirements.txt`` into a SEPARATE venv.
2. Download a model (``G_*.pth``) + matching ``config.json``, plus the
   ``bert/`` and ``emotional/clap-htsat-fused`` weights (see Bert-VITS2 README).
3. From the Bert-VITS2 repo root, inside that venv::

       set BV_REPO=C:\\path\\to\\Bert-VITS2
       set BV_MODEL=C:\\path\\to\\G_xxx.pth
       set BV_CONFIG=C:\\path\\to\\config.json
       python C:\\path\\to\\VoiceStudio\\engines\\bert_vits2_server.py

4. In VoiceStudio, set ``BERT_VITS2_URL=http://127.0.0.1:5000/tts`` (and
   optionally ``BERT_VITS2_SPEAKER``), then pick the "Bert-VITS2 (HTTP bridge)"
   engine in Studio.

Environment: BV_REPO (default "."), BV_MODEL, BV_CONFIG, BV_DEVICE,
BV_HOST (default 127.0.0.1), BV_PORT (default 5000).

NOTE: This file is a reference adapter for the Bert-VITS2 environment. It is not
imported by VoiceStudio and is not covered by VoiceStudio's dependency pins.
"""
import base64
import io
import os
import re
import sys
from pathlib import Path

# Bert-VITS2 needs both its repo root (utils/config/models/text) and the
# for_deploy dir (infer/infer_utils) on the path.
BV_REPO = Path(os.environ.get("BV_REPO", ".")).resolve()
sys.path.insert(0, str(BV_REPO))
sys.path.insert(0, str(BV_REPO / "for_deploy"))

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

import utils
from infer import get_net_g, infer, latest_version
from infer_utils import BertFeature, ClapFeature

DEVICE = os.environ.get(
    "BV_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL = os.environ.get("BV_MODEL", "")
CONFIG = os.environ.get("BV_CONFIG", "")

_state: dict = {}


def _load() -> None:
    hps = utils.get_hparams_from_file(CONFIG)
    version = getattr(hps, "version", latest_version)
    net_g = get_net_g(model_path=MODEL, version=version, device=DEVICE, hps=hps)
    bert = {
        "ZH": BertFeature("./bert/chinese-roberta-wwm-ext-large", language="ZH"),
        "JP": BertFeature("./bert/deberta-v2-large-japanese-char-wwm", language="JP"),
        "EN": BertFeature("./bert/deberta-v3-large", language="EN"),
    }
    clap = ClapFeature("./emotional/clap-htsat-fused")
    _state.update(
        hps=hps, net_g=net_g, bert=bert, clap=clap,
        speakers=list(hps.data.spk2id.keys()),
        sr=int(hps.data.sampling_rate),
    )
    print(f"[bert-vits2] loaded {MODEL} on {DEVICE}; "
          f"speakers={_state['speakers']}")


def _split(text: str) -> list[str]:
    """Split into sentence-ish pieces so long text doesn't OOM the decoder."""
    parts = re.split(r"(?<=[。！？!?.\n])", text)
    return [p.strip() for p in parts if p and p.strip()]


class TTSReq(BaseModel):
    text: str
    speaker: str = ""
    language: str = "auto"
    sdp_ratio: float = 0.2
    noise_scale: float = 0.6
    noise_scale_w: float = 0.8
    length_scale: float = 1.0
    emotion: str = "Happy"
    reference_audio_b64: str | None = None


app = FastAPI(title="Bert-VITS2 adapter", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    _load()


@app.get("/speakers")
def speakers():
    return {"speakers": _state.get("speakers", [])}


@app.post("/tts")
def tts(req: TTSReq):
    hps, net_g = _state["hps"], _state["net_g"]
    speaker = req.speaker or (_state["speakers"][0] if _state["speakers"] else "")

    ref = None
    if req.reference_audio_b64:
        import librosa
        raw = base64.b64decode(req.reference_audio_b64)
        ref, _ = librosa.load(io.BytesIO(raw), sr=_state["sr"], mono=True)

    pieces = _split(req.text)
    if not pieces:
        return Response(status_code=400, content="Empty text.")

    sr = _state["sr"]
    gap = np.zeros(int(sr * 0.15), dtype=np.float32)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for piece in pieces:
            audio = infer(
                piece,
                reference_audio=ref,
                emotion=req.emotion,
                sdp_ratio=req.sdp_ratio,
                noise_scale=req.noise_scale,
                noise_scale_w=req.noise_scale_w,
                length_scale=req.length_scale,
                sid=speaker,
                language=req.language,
                hps=hps,
                net_g=net_g,
                device=DEVICE,
                bert=_state["bert"],
                clap=_state["clap"],
            )
            chunks.append(np.asarray(audio, dtype=np.float32))
            chunks.append(gap)

    wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    if not MODEL or not CONFIG:
        sys.exit("Set BV_MODEL and BV_CONFIG (paths to G_*.pth and config.json).")
    uvicorn.run(app,
                host=os.environ.get("BV_HOST", "127.0.0.1"),
                port=int(os.environ.get("BV_PORT", "5000")),
                log_level="info")
