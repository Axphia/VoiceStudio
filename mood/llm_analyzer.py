"""LLM-based Mood and Prosody Analyzer.

Interacts with any OpenAI-compatible LLM endpoint (Ollama, LM Studio, vLLM, Qwen,
OpenAI, Groq, Gemini) to extract nuanced emotional cues, character subtext,
and TTS speech prompts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .models import EmotionScore, MoodResult, ProsodySettings, TTSStyleInstruction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Voice Acting Director and Emotional Prosody Specialist for advanced Text-to-Speech synthesis.
Your job is to analyze the emotional mood, psychological subtext, and delivery style of spoken text.

You MUST respond strictly with valid JSON conforming to this schema:
{
  "primary_emotion": "happy|sad|angry|fearful|surprised|disgusted|exhausted|whisper|tender|neutral",
  "intensity": 0.0 to 1.0,
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "subtext": "brief explanation of underlying emotion or acting cue",
  "prosody": {
    "pitch_shift_semitones": -4.0 to 4.0,
    "speed_ratio": 0.6 to 1.4,
    "energy_ratio": 0.5 to 1.5,
    "breathiness": 0.0 to 1.0,
    "pause_scale": 0.8 to 2.0
  },
  "tts_style": {
    "style_prompt": "description of voice delivery (e.g., 'weary, strained sigh, speaking with heavy exhaustion')",
    "emotion_tags": ["tired", "sigh"],
    "fish_audio_text": "[tired] Text here..."
  }
}
Do not include any commentary or markdown codeblocks outside the JSON.
"""

SCRIPT_SYSTEM_PROMPT = """You are an expert Voice Acting Director.
Analyze the following multi-character script line by line.
For each spoken line, determine the speaker, their primary emotion, intensity (0.0 to 1.0), prosody instructions, subtext, and Fish Audio inline tagged text (e.g. [tired], [concerned], [whisper]).

Respond strictly with valid JSON:
{
  "dialogue": [
    {
      "speaker": "Speaker Name",
      "text": "The line text",
      "primary_emotion": "exhausted|tender|etc.",
      "intensity": 0.75,
      "valence": -0.6,
      "arousal": 0.3,
      "subtext": "...",
      "prosody": {
        "pitch_shift_semitones": -1.5,
        "speed_ratio": 0.85,
        "energy_ratio": 0.75,
        "breathiness": 0.4,
        "pause_scale": 1.3
      },
      "fish_audio_text": "[tired] ..."
    }
  ]
}
"""


class LLMMoodAnalyzer:
    """Performs advanced emotion & prosody analysis using an LLM."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        # Default to local Ollama if present or configured via env
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or "http://127.0.0.1:11434/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL") or "qwen2.5:7b"
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if the LLM endpoint is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _call_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """Sends chat completion request to the OpenAI-compatible endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]

    def analyze(self, text: str) -> MoodResult:
        """Analyzes a single text string using the LLM."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text:\n\n{cleaned_text}"}
        ]

        raw_output = self._call_chat_completion(messages)
        return self._parse_json_result(cleaned_text, raw_output)

    def analyze_dialogue_script(self, script_text: str) -> List[MoodResult]:
        """Analyzes a multi-turn conversation or script."""
        cleaned_script = (script_text or "").strip()
        if not cleaned_script:
            return []

        messages = [
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this script:\n\n{cleaned_script}"}
        ]

        raw_output = self._call_chat_completion(messages)
        
        # Clean potential markdown fences
        clean_json = re.sub(r"^```(?:json)?\s*", "", raw_output.strip())
        clean_json = re.sub(r"\s*```$", "", clean_json)

        try:
            parsed = json.loads(clean_json)
            dialogue_items = parsed.get("dialogue", [])
            results = []
            for item in dialogue_items:
                prosody_data = item.get("prosody", {})
                prosody = ProsodySettings(
                    pitch_shift_semitones=float(prosody_data.get("pitch_shift_semitones", 0.0)),
                    speed_ratio=float(prosody_data.get("speed_ratio", 1.0)),
                    energy_ratio=float(prosody_data.get("energy_ratio", 1.0)),
                    breathiness=float(prosody_data.get("breathiness", 0.0)),
                    pause_scale=float(prosody_data.get("pause_scale", 1.0)),
                )
                fish_text = item.get("fish_audio_text") or item.get("text", "")
                tts_style = TTSStyleInstruction(
                    style_prompt=item.get("subtext", ""),
                    emotion_tags=[item.get("primary_emotion", "neutral")],
                    fish_audio_text=fish_text,
                )
                res = MoodResult(
                    text=item.get("text", ""),
                    primary_emotion=item.get("primary_emotion", "neutral"),
                    confidence=0.95,
                    intensity=float(item.get("intensity", 0.5)),
                    valence=float(item.get("valence", 0.0)),
                    arousal=float(item.get("arousal", 0.5)),
                    prosody=prosody,
                    tts_style=tts_style,
                    subtext=item.get("subtext", ""),
                    speaker=item.get("speaker"),
                    source="llm"
                )
                results.append(res)
            return results
        except Exception as e:
            logger.error("Failed to parse dialogue script JSON from LLM: %s. Output: %s", e, raw_output)
            raise

    def _parse_json_result(self, original_text: str, json_text: str) -> MoodResult:
        """Parses LLM JSON response into MoodResult."""
        clean_json = re.sub(r"^```(?:json)?\s*", "", json_text.strip())
        clean_json = re.sub(r"\s*```$", "", clean_json)

        data = json.loads(clean_json)
        prosody_data = data.get("prosody", {})
        tts_data = data.get("tts_style", {})

        prosody = ProsodySettings(
            pitch_shift_semitones=float(prosody_data.get("pitch_shift_semitones", 0.0)),
            speed_ratio=float(prosody_data.get("speed_ratio", 1.0)),
            energy_ratio=float(prosody_data.get("energy_ratio", 1.0)),
            breathiness=float(prosody_data.get("breathiness", 0.0)),
            pause_scale=float(prosody_data.get("pause_scale", 1.0)),
        )

        tts_style = TTSStyleInstruction(
            style_prompt=str(tts_data.get("style_prompt", "")),
            emotion_tags=list(tts_data.get("emotion_tags", [])),
            fish_audio_text=str(tts_data.get("fish_audio_text", original_text)),
            bert_vits2_emotion=str(data.get("primary_emotion", "neutral")),
        )

        return MoodResult(
            text=original_text,
            primary_emotion=str(data.get("primary_emotion", "neutral")),
            confidence=0.95,
            intensity=float(data.get("intensity", 0.5)),
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.5)),
            emotions=[EmotionScore(str(data.get("primary_emotion", "neutral")), 0.95)],
            prosody=prosody,
            tts_style=tts_style,
            subtext=str(data.get("subtext", "")),
            source="llm"
        )
