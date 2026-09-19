"""Pluggable TTS engines (ElevenLabs-style)."""
from .base import VoiceEngine
from .registry import get_engine, list_engines, register

__all__ = ["VoiceEngine", "get_engine", "list_engines", "register"]
