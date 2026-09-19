"""Built-in Coqui XTTS-v2 engine (default)."""
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch

from .base import VoiceEngine
from .registry import register


@register
class XTTSEngine(VoiceEngine):
    name = "xtts"
    label = "Coqui XTTS-v2 (local, CUDA)"
    needs_ref = True

    def __init__(self):
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
            from TTS.api import TTS

            t0 = time.perf_counter()
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"[xtts] model load: {time.perf_counter() - t0:.1f}s")
        return self._model

    def synthesize(self, text, out, language="en", ref_audio=None, **options) -> Path:
        if ref_audio is None:
            raise ValueError("XTTS needs ref_audio (a reference voice WAV).")
        model = self._model_lazy()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        model.tts_to_file(
            text=text.strip(),
            speaker_wav=str(ref_audio),
            language=language,
            file_path=str(out),
            speed=options.get("speed", 1.0),
            split_sentences=options.get("split_sentences", True),
        )
        print(f"[xtts] [{time.perf_counter() - t0:.1f}s] Saved -> {out}")
        return out
