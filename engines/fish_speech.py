"""Fish Speech (Fish Audio S1/S2) engine via HTTP bridge to the official server.

Fish Speech is the model behind fish.audio: a Dual-AR LLM TTS that clones a voice
**zero-shot** from a short reference, in 80+ languages, with inline emotion /
prosody tags such as ``[whisper]``, ``[excited]``, ``[angry]`` typed straight
into the text. It ships its own API server (``tools/api_server.py`` →
``POST /v1/tts``), so this engine is a thin client: no custom adapter and no
dependency conflicts with XTTS.

Config (environment variables, or per-call ``**options``):
  FISH_SPEECH_URL       default http://127.0.0.1:8080/v1/tts
  FISH_SPEECH_TOKEN     optional bearer token (server started with an api key)
  FISH_SPEECH_REF_ID    optional server-stored reference_id (fish.audio/m/<id>)
  FISH_SPEECH_REF_TEXT  transcript of the reference voice (improves cloning)
  FISH_SPEECH_TIMEOUT   request seconds (default 180)

Note: Fish Speech is multilingual and infers language from the text + reference,
so the VoiceStudio ``language`` argument is accepted but not sent.
"""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .base import VoiceEngine
from .registry import register


@register
class FishSpeechEngine(VoiceEngine):
    name = "fish-speech"
    label = "Fish Speech S1/S2 (HTTP bridge)"
    needs_ref = True             # zero-shot: clones the Studio reference voice
    supports_direct = True       # or speak with a stored reference_id, no upload
    supports_zero_shot = True

    def synthesize(self, text, out, language="en", ref_audio=None, **options) -> Path:
        url = options.get("url") or os.environ.get(
            "FISH_SPEECH_URL", "http://127.0.0.1:8080/v1/tts")
        token = options.get("token") or os.environ.get("FISH_SPEECH_TOKEN")
        ref_id = options.get("reference_id") or os.environ.get("FISH_SPEECH_REF_ID")
        ref_text = options.get("ref_text") or os.environ.get("FISH_SPEECH_REF_TEXT", "")
        timeout = float(options.get("timeout")
                        or os.environ.get("FISH_SPEECH_TIMEOUT", "180"))

        text = (text or "").strip()
        if not text:
            raise ValueError("Fish Speech needs non-empty text.")

        # Defaults stay inside the server's validation ranges
        # (temperature/top_p 0.1-1.0, repetition_penalty 0.9-2.0, chunk 100-1000).
        payload = {
            "text": text,
            "format": "wav",
            "streaming": False,
            "normalize": True,
            "chunk_length": int(options.get("chunk_length", 200)),
            "temperature": float(options.get("temperature", 0.8)),
            "top_p": float(options.get("top_p", 0.8)),
            "repetition_penalty": float(options.get("repetition_penalty", 1.1)),
            "max_new_tokens": int(options.get("max_new_tokens", 1024)),
        }
        if ref_id:
            payload["reference_id"] = ref_id
        if ref_audio is not None:
            ref = Path(ref_audio)
            if ref.is_file():
                payload["references"] = [{
                    "audio": base64.b64encode(ref.read_bytes()).decode(),
                    "text": ref_text,
                }]

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Fish Speech server error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Fish Speech server at {url} ({e.reason}). Start it "
                f"in your fish-speech venv:  python -m tools.api_server") from e

        if not body:
            raise RuntimeError("Fish Speech returned empty audio.")
        out.write_bytes(body)
        return out
