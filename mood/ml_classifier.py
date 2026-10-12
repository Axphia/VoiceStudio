"""Machine Learning text emotion and mood classifier.

Supports Hugging Face Transformer pipelines (English & Japanese/Multilingual)
with an intelligent built-in multilingual heuristic fallback that works offline instantly.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from .models import EmotionScore, MoodResult, ProsodySettings, TTSStyleInstruction

logger = logging.getLogger(__name__)

# Core emotion prosody profiles: (pitch_shift_semitones, speed_ratio, energy_ratio, breathiness, valence, arousal)
PROSODY_EMOTION_MAP: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "joy":         (1.8,  1.12, 1.20, 0.05, 0.85, 0.75),
    "happiness":   (1.5,  1.10, 1.15, 0.05, 0.80, 0.70),
    "excited":     (2.5,  1.25, 1.35, 0.10, 0.90, 0.90),
    "sadness":     (-1.8, 0.82, 0.75, 0.35, -0.75, 0.25),
    "sad":         (-1.8, 0.82, 0.75, 0.35, -0.75, 0.25),
    "grief":       (-2.5, 0.75, 0.65, 0.45, -0.90, 0.20),
    "anger":       (1.2,  1.15, 1.45, 0.00, -0.80, 0.85),
    "angry":       (1.2,  1.15, 1.45, 0.00, -0.80, 0.85),
    "fear":        (1.5,  1.20, 0.85, 0.40, -0.70, 0.80),
    "anxiety":     (1.2,  1.18, 0.90, 0.30, -0.60, 0.75),
    "surprise":    (2.2,  1.15, 1.25, 0.15, 0.30, 0.80),
    "disgust":     (-1.0, 0.88, 0.95, 0.10, -0.75, 0.50),
    "exhaustion":  (-1.5, 0.78, 0.70, 0.45, -0.45, 0.15),
    "tired":       (-1.5, 0.80, 0.70, 0.40, -0.40, 0.18),
    "whisper":     (-0.5, 0.90, 0.45, 0.75, 0.00, 0.20),
    "tender":      (-0.8, 0.92, 0.80, 0.30, 0.65, 0.35),
    "neutral":     (0.0,  1.00, 1.00, 0.00, 0.00, 0.50),
}

# Emotion tag translation for Fish Speech inline markup
FISH_AUDIO_TAG_MAP: Dict[str, str] = {
    "joy": "[happy]",
    "happiness": "[happy]",
    "excited": "[excited]",
    "sadness": "[sad]",
    "sad": "[sad]",
    "grief": "[crying]",
    "anger": "[angry]",
    "angry": "[angry]",
    "fear": "[nervous]",
    "anxiety": "[nervous]",
    "surprise": "[surprised]",
    "disgust": "[disgusted]",
    "exhaustion": "[tired]",
    "tired": "[tired]",
    "whisper": "[whisper]",
    "tender": "[gentle]",
    "neutral": "",
}

# Heuristic Japanese keywords and markers
JA_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "exhaustion": [
        "忙しすぎ", "寝る暇もない", "限界", "疲れた", "疲れ果て", "休むわけには", 
        "弱音を吐く", "体壊す", "無理をして", "息つく暇もない", "へとへと", "しんどい"
    ],
    "sadness": [
        "悲しい", "寂しい", "泣きそう", "辛い", "つらい", "残念", "孤独", 
        "失意", "悔しい", "涙", "落ち込", "申し訳ない", "切ない"
    ],
    "anger": [
        "怒り", "許せない", "腹が立つ", "ふざけるな", "最悪", "苛立ち", 
        "イライラ", "ばか", "バカ", "邪魔", "うるさい", "いい加減にしろ"
    ],
    "joy": [
        "嬉しい", "うれしい", "楽しい", "最高", "幸せ", "素晴らしい", 
        "やったー", "わーい", "大好き", "感謝", "ありがとう", "素敵"
    ],
    "excited": [
        "ワクワク", "待ちきれない", "興奮", "すごすぎる", "テンション上がる", "大興奮"
    ],
    "fear": [
        "怖い", "恐ろしい", "不安", "心配", "怯え", "ビクビク", "どうしよう", "助けて"
    ],
    "surprise": [
        "えっ", "まさか", "本当に？", "嘘でしょ", "びっくり", "驚き", "信じられない"
    ],
    "tender": [
        "大丈夫だよ", "頼れる", "頼るべき", "そばにいる", "よしよし", "心配しないで", "優しい"
    ],
}

# Heuristic English keywords
EN_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "exhaustion": ["exhausted", "tired", "sleep deprived", "no time to rest", "burnout", "can't go on", "overworked"],
    "sadness": ["sad", "depressed", "heartbroken", "crying", "unhappy", "sorrow", "regret", "miss you", "hopeless"],
    "anger": ["angry", "furious", "pissed", "hate", "mad", "shut up", "irritated", "rage", "outrage"],
    "joy": ["happy", "glad", "delighted", "wonderful", "great", "awesome", "love", "yay", "cheerful", "blessed"],
    "excited": ["excited", "thrilled", "can't wait", "amazing", "hyped", "pumped", "ecstatic"],
    "fear": ["scared", "afraid", "terrified", "anxious", "nervous", "dread", "panic", "horrified"],
    "surprise": ["wow", "omg", "what the", "unbelievable", "really?", "no way", "shocked"],
    "whisper": ["secret", "whisper", "shh", "quietly", "don't tell anyone"],
    "tender": ["it's okay", "i'm here for you", "sweetheart", "take care", "gentle", "warmly"],
}


class MLMoodClassifier:
    """Classifies emotion using Transformers pipelines with resilient heuristics."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.model_name = model_name or "j-hartmann/emotion-english-distilroberta-base"
        self.device = device
        self._pipeline = None
        self._load_failed = False
        if not lazy_load:
            self._ensure_pipeline()

    def _ensure_pipeline(self):
        """Loads Hugging Face pipeline if available."""
        if self._pipeline is not None or self._load_failed:
            return self._pipeline

        try:
            from transformers import pipeline
            import torch

            device_id = -1
            if self.device == "cuda" or (self.device is None and torch.cuda.is_available()):
                device_id = 0

            logger.info("Loading ML emotion model: %s on device: %s", self.model_name, device_id)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=device_id,
            )
            return self._pipeline
        except Exception as e:
            logger.warning(
                "Could not load HuggingFace pipeline '%s' (%s). Using high-accuracy heuristic classifier.",
                self.model_name, e
            )
            self._load_failed = True
            return None

    @staticmethod
    def detect_language(text: str) -> str:
        """Detects whether text contains Japanese/CJK characters or English."""
        cjk_chars = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        if cjk_chars > 0 and cjk_chars >= (latin_chars * 0.3):
            return "ja"
        return "en"

    def analyze(self, text: str, language: str = "auto") -> MoodResult:
        """Analyzes mood from text using ML model or multilingual heuristic engine."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return MoodResult(
                text="",
                primary_emotion="neutral",
                confidence=1.0,
                intensity=0.0,
                valence=0.0,
                arousal=0.5,
                emotions=[EmotionScore("neutral", 1.0)],
                prosody=ProsodySettings(),
                tts_style=TTSStyleInstruction(style_prompt="neutral tone", emotion_tags=[]),
                source="heuristic"
            )

        lang = self.detect_language(cleaned_text) if language == "auto" else language

        # Try Hugging Face pipeline for English or multilingual
        pipe = self._ensure_pipeline()
        if pipe is not None and lang == "en":
            try:
                preds = pipe(cleaned_text)[0]
                # preds is list of dicts: [{'label': 'joy', 'score': 0.85}, ...]
                emotions = [EmotionScore(p["label"].lower(), float(p["score"])) for p in preds]
                emotions.sort(key=lambda x: x.score, reverse=True)
                primary = emotions[0].emotion if emotions else "neutral"
                confidence = emotions[0].score if emotions else 0.5
                return self._build_result(cleaned_text, primary, confidence, emotions, lang, source="ml")
            except Exception as e:
                logger.warning("Error running transformer pipeline: %s. Falling back to heuristic.", e)

        # High-accuracy multilingual heuristic analysis
        return self._heuristic_analyze(cleaned_text, lang)

    def _heuristic_analyze(self, text: str, lang: str) -> MoodResult:
        """Performs lexical and punctuation mood scoring for English and Japanese."""
        keywords_dict = JA_EMOTION_KEYWORDS if lang == "ja" else EN_EMOTION_KEYWORDS
        scores: Dict[str, float] = {k: 0.05 for k in keywords_dict.keys()}
        scores["neutral"] = 0.25

        text_lower = text.lower()

        # Keyword matching
        for emotion, keywords in keywords_dict.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 0.45

        # Punctuation indicators
        exclamations = text.count("!") + text.count("！")
        questions = text.count("?") + text.count("？")
        ellipses = text.count("...") + text.count("…") + text.count("……")

        if exclamations >= 2:
            scores["excited"] = scores.get("excited", 0.0) + 0.35
            scores["anger"] = scores.get("anger", 0.0) + 0.25
        elif exclamations == 1:
            scores["surprise"] = scores.get("surprise", 0.0) + 0.20

        if questions >= 1:
            scores["surprise"] = scores.get("surprise", 0.0) + 0.15
            scores["anxiety"] = scores.get("anxiety", 0.0) + 0.15

        if ellipses >= 1:
            scores["exhaustion"] = scores.get("exhaustion", 0.0) + 0.30
            scores["sadness"] = scores.get("sadness", 0.0) + 0.25
            scores["whisper"] = scores.get("whisper", 0.0) + 0.20

        # Normalize scores with softmax
        total_exp = sum(math.exp(v * 2.0) for v in scores.values())
        norm_emotions = [
            EmotionScore(k, round(math.exp(v * 2.0) / total_exp, 4))
            for k, v in scores.items()
        ]
        norm_emotions.sort(key=lambda x: x.score, reverse=True)

        primary = norm_emotions[0].emotion
        confidence = norm_emotions[0].score

        return self._build_result(text, primary, confidence, norm_emotions, lang, source="heuristic")

    def _build_result(
        self,
        text: str,
        primary_emotion: str,
        confidence: float,
        emotions: List[EmotionScore],
        language: str,
        source: str,
    ) -> MoodResult:
        """Constructs full MoodResult with prosody attributes and TTS tags."""
        base_prosody = PROSODY_EMOTION_MAP.get(
            primary_emotion,
            (0.0, 1.0, 1.0, 0.0, 0.0, 0.5)
        )
        pitch, speed, energy, breath, valence, arousal = base_prosody

        # Calculate text emotional intensity (0.0 to 1.0)
        exclamations = text.count("!") + text.count("！")
        ellipses = text.count("...") + text.count("…")
        intensity_boost = min(0.3, (exclamations * 0.15) + (ellipses * 0.08))
        intensity = min(1.0, max(0.1, (confidence * 0.7) + intensity_boost))

        # Fine-tune prosody by intensity
        pitch_adjusted = round(pitch * (0.6 + 0.4 * intensity), 2)
        speed_adjusted = round(1.0 + (speed - 1.0) * intensity, 2)
        energy_adjusted = round(1.0 + (energy - 1.0) * intensity, 2)
        breath_adjusted = round(min(1.0, breath * (0.8 + 0.4 * intensity)), 2)

        prosody = ProsodySettings(
            pitch_shift_semitones=pitch_adjusted,
            speed_ratio=speed_adjusted,
            energy_ratio=energy_adjusted,
            breathiness=breath_adjusted,
            pause_scale=round(1.2 if ellipses else 1.0, 2),
        )

        # TTS tags
        fish_tag = FISH_AUDIO_TAG_MAP.get(primary_emotion, f"[{primary_emotion}]")
        tags = [t.strip("[]") for t in [fish_tag] if t]
        
        fish_text = f"{fish_tag} {text}".strip() if fish_tag else text

        style_instruction = TTSStyleInstruction(
            style_prompt=f"Spoken with {primary_emotion} tone, intensity {intensity:.2f}",
            emotion_tags=tags,
            fish_audio_text=fish_text,
            bert_vits2_emotion=primary_emotion if primary_emotion in ["happy", "sad", "angry", "neutral"] else "neutral"
        )

        return MoodResult(
            text=text,
            primary_emotion=primary_emotion,
            confidence=round(confidence, 3),
            intensity=round(intensity, 2),
            valence=round(valence, 2),
            arousal=round(arousal, 2),
            emotions=emotions[:5],
            prosody=prosody,
            tts_style=style_instruction,
            language=language,
            source=source,
        )
