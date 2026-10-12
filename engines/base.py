"""VoiceStudio engine interface. Implement this to add a new TTS backend."""
from abc import ABC, abstractmethod
from pathlib import Path


class VoiceEngine(ABC):
    """A swappable TTS backend (ElevenLabs-style)."""

    #: unique id used with --engine / registry, e.g. "xtts", "f5"
    name: str = "base"
    #: human label for UIs
    label: str = "Base engine"
    #: True if the engine needs a reference voice file (voice cloning)
    needs_ref: bool = True
    #: True if the engine supports text-to-speech directly without reference
    supports_direct: bool = False
    #: True if the engine supports zero-shot voice cloning
    supports_zero_shot: bool = False

    @abstractmethod
    def synthesize(
        self,
        text: str,
        out: Path,
        language: str = "en",
        ref_audio: Path | None = None,
        **options,
    ) -> Path:
        """Synthesize text to a WAV file. Returns the output path."""
        raise NotImplementedError
