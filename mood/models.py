"""Data models for text mood, emotion, and TTS prosody analysis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmotionScore:
    """Individual emotion probability or weight."""
    emotion: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProsodySettings:
    """Speech synthesis prosody and delivery attributes."""
    pitch_shift_semitones: float = 0.0  # e.g., +2.0 (higher/excited) or -2.0 (somber/serious)
    speed_ratio: float = 1.0            # e.g., 1.15 (fast/anxious/excited) or 0.85 (slow/sad/tired)
    energy_ratio: float = 1.0           # e.g., 1.2 (intense/angry/joy) or 0.7 (depressed/whispering)
    breathiness: float = 0.0            # 0.0 to 1.0 (higher for whispers, sighs, exhaustion)
    pause_scale: float = 1.0            # multiplier for punctuation pauses

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_ssml_attributes(self) -> str:
        """Returns standard SSML prosody tag attributes."""
        pitch_pct = int(self.pitch_shift_semitones * 5)
        rate_pct = int((self.speed_ratio - 1.0) * 100)
        volume_pct = int((self.energy_ratio - 1.0) * 100)
        pitch_str = f"{'+' if pitch_pct >= 0 else ''}{pitch_pct}%"
        rate_str = f"{'+' if rate_pct >= 0 else ''}{rate_pct}%"
        volume_str = f"{'+' if volume_pct >= 0 else ''}{volume_pct}%"
        return f'pitch="{pitch_str}" rate="{rate_str}" volume="{volume_str}"'


@dataclass
class TTSStyleInstruction:
    """TTS formatting and engine-specific cues."""
    style_prompt: str = ""                         # Natural language prompt e.g. "spoken with tired, strained exhaustion"
    emotion_tags: List[str] = field(default_factory=list) # e.g. ["whisper", "sad", "exhausted"]
    fish_audio_text: str = ""                      # Text with inline [tag] for Fish Speech, e.g. "[exhausted] 最近、仕事が..."
    bert_vits2_emotion: str = "neutral"            # Discrete Bert-VITS2 emotion (neutral, happy, sad, angry, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MoodResult:
    """Comprehensive result of mood analysis on a text snippet."""
    text: str
    primary_emotion: str
    confidence: float
    intensity: float                     # 0.0 (subtle) to 1.0 (extreme/screaming/crying)
    valence: float                       # -1.0 (very negative) to +1.0 (very positive)
    arousal: float                       # 0.0 (calm/lethargic) to 1.0 (hyper-aroused/frantic)
    emotions: List[EmotionScore] = field(default_factory=list)
    prosody: ProsodySettings = field(default_factory=ProsodySettings)
    tts_style: TTSStyleInstruction = field(default_factory=TTSStyleInstruction)
    subtext: str = ""                    # Underlying emotional intent or acting cue
    speaker: Optional[str] = None        # Detected character/speaker name if in script format
    language: str = "en"
    source: str = "ml"                   # "ml", "llm", or "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def format_for_fish_speech(self) -> str:
        """Returns text prepared for Fish Speech with inline emotion tags."""
        if self.tts_style and self.tts_style.fish_audio_text:
            return self.tts_style.fish_audio_text
        tags = " ".join(f"[{t.strip('[]')}]" for t in self.tts_style.emotion_tags)
        return f"{tags} {self.text}".strip() if tags else self.text
