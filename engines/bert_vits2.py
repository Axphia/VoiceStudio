"""Bert-VITS2 engine via HTTP bridge.

Bert-VITS2 speaks with a *trained speaker* (``sid``) rather than zero-shot
cloning, and its dependency stack (librosa 0.9.2, gradio 3.50, its own
transformers) conflicts with the XTTS pins. So we do NOT import it here.
Instead this engine POSTs to a Bert-VITS2 adapter server you run separately
(see ``engines/bert_vits2_server.py``) and saves the WAV it returns.

Config (environment variables, or per-call ``**options``):
  BERT_VITS2_URL      adapter endpoint     (default http://127.0.0.1:5000/tts)
  BERT_VITS2_SPEAKER  default speaker name (default "" = adapter's first speaker)
  BERT_VITS2_TIMEOUT  request seconds      (default 120)
"""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .base import VoiceEngine
from .registry import register

# VoiceStudio language code -> Bert-VITS2 language code.
_LANG = {"ja": "JP", "zh": "ZH", "en": "EN"}


@register
class BertVITS2Engine(VoiceEngine):
    name = "bert-vits2"
    label = "Bert-VITS2 (HTTP bridge)"
    needs_ref = False            # reference audio is optional (emotion only)
    supports_direct = True       # speaks in a trained voice without a clone ref
    supports_zero_shot = False   # speakers are baked into the model

    def synthesize(self, text, out, language="en", ref_audio=None, **options) -> Path:
        url = options.get("url") or os.environ.get(
            "BERT_VITS2_URL", "http://127.0.0.1:5000/tts")
        speaker = options.get("speaker") or os.environ.get("BERT_VITS2_SPEAKER", "")
        timeout = float(options.get("timeout")
                        or os.environ.get("BERT_VITS2_TIMEOUT", "120"))
        lang = _LANG.get((language or "en").lower(), "auto")

        payload = {
            "text": (text or "").strip(),
            "speaker": speaker,
            "language": lang,
            "sdp_ratio": options.get("sdp_ratio", 0.2),
            "noise_scale": options.get("noise_scale", 0.6),
            "noise_scale_w": options.get("noise_scale_w", 0.8),
            "length_scale": options.get("length_scale", 1.0),
            "emotion": options.get("emotion", "Happy"),
        }
        if not payload["text"]:
            raise ValueError("Bert-VITS2 needs non-empty text.")

        if ref_audio is not None:
            ref = Path(ref_audio)
            if ref.is_file():
                payload["reference_audio_b64"] = base64.b64encode(
                    ref.read_bytes()).decode()

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Bert-VITS2 server error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Bert-VITS2 adapter at {url} ({e.reason}). Start "
                f"engines/bert_vits2_server.py inside your Bert-VITS2 venv.") from e

        if "application/json" in ctype:  # adapter may return {"audio_b64": ...}
            data = json.loads(body)
            if "audio_b64" in data:
                body = base64.b64decode(data["audio_b64"])
            else:
                raise RuntimeError(
                    f"Unexpected JSON from Bert-VITS2 adapter: {str(data)[:200]}")
        if not body:
            raise RuntimeError("Bert-VITS2 adapter returned empty audio.")

        out.write_bytes(body)
        return out
