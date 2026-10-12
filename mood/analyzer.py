"""Unified Mood and Prosody Analyzer for VoiceStudio.

Combines Machine Learning classifiers and Large Language Models into a single,
clean API with auto-fallback, multilingual dialogue parsing, and TTS tagging.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .llm_analyzer import LLMMoodAnalyzer
from .ml_classifier import MLMoodClassifier
from .models import EmotionScore, MoodResult, ProsodySettings, TTSStyleInstruction

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """High-level unified API for text emotion, mood, and TTS prosody."""

    def __init__(
        self,
        ml_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        prefer_llm: bool = False,
    ):
        self.ml = MLMoodClassifier(model_name=ml_model)
        self.llm = LLMMoodAnalyzer(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
        )
        self.prefer_llm = prefer_llm

    def analyze(
        self,
        text: str,
        mode: str = "auto",
        language: str = "auto",
    ) -> MoodResult:
        """Analyzes text mood.
        
        Args:
            text: Input sentence or line.
            mode: "auto", "ml", or "llm".
            language: "auto", "en", or "ja".
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return self.ml.analyze("", language=language)

        # Determine strategy
        use_llm = False
        if mode == "llm":
            use_llm = True
        elif mode == "auto" and self.prefer_llm:
            use_llm = self.llm.is_available()

        if use_llm:
            try:
                return self.llm.analyze(cleaned)
            except Exception as e:
                logger.warning("LLM mood analysis failed (%s). Falling back to ML.", e)

        return self.ml.analyze(cleaned, language=language)

    def analyze_dialogue(
        self,
        script_text: str,
        mode: str = "auto",
    ) -> List[MoodResult]:
        """Analyzes an entire multi-character dialogue script.
        
        Supports formats:
        - Japanese style: 田中「最近、仕事が忙しすぎて...」
        - Western script style: Tanaka: Lately work has been so busy...
        """
        lines = [line.strip() for line in script_text.strip().splitlines() if line.strip()]
        if not lines:
            return []

        # Try LLM dialogue extraction if available and requested
        if (mode == "llm" or (mode == "auto" and self.prefer_llm)) and self.llm.is_available():
            try:
                results = self.llm.analyze_dialogue_script(script_text)
                if results:
                    return results
            except Exception as e:
                logger.warning("LLM dialogue analysis failed (%s). Falling back to line-by-line parsing.", e)

        # Line-by-line parsing with speaker extraction
        parsed_results: List[MoodResult] = []
        for line in lines:
            speaker = None
            dialogue_text = line

            # Match Japanese dialogue: 田中「...」
            ja_match = re.match(r"^([^「]+)「(.*?)」?$", line)
            if ja_match:
                speaker = ja_match.group(1).strip()
                dialogue_text = ja_match.group(2).strip()
            else:
                # Match Western style: Name: ...
                western_match = re.match(r"^([A-Za-z0-9_\u4e00-\u9faf\u3040-\u30ff]+)\s*[:：]\s*(.*)$", line)
                if western_match:
                    speaker = western_match.group(1).strip()
                    dialogue_text = western_match.group(2).strip()

            result = self.analyze(dialogue_text, mode=mode)
            result.speaker = speaker
            parsed_results.append(result)

        return parsed_results

    def format_tts_script(
        self,
        script_text: str,
        engine: str = "fish-speech",
    ) -> str:
        """Enriches raw script text with inline emotion tags suitable for TTS."""
        results = self.analyze_dialogue(script_text)
        formatted_lines: List[str] = []

        for r in results:
            tagged_content = r.format_for_fish_speech() if engine == "fish-speech" else r.text
            if r.speaker:
                if "「" in script_text or "」" in script_text:
                    formatted_lines.append(f"{r.speaker}「{tagged_content}」")
                else:
                    formatted_lines.append(f"{r.speaker}: {tagged_content}")
            else:
                formatted_lines.append(tagged_content)

        return "\n".join(formatted_lines)
