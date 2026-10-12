#!/usr/bin/env python3
"""Test and synthesize speech with mood and emotional prosody modulation.

Examples:
  # 1. Synthesize a single sentence with auto-detected mood:
  python scripts/test_mood_sound.py --text "最近、仕事が忙しすぎて、寝る暇もないよ。" --language ja

  # 2. Run a mood comparison showcase (Happy, Sad, Angry, Exhausted):
  python scripts/test_mood_sound.py --preset showcase

  # 3. Test a dialogue from a script file:
  python scripts/test_mood_sound.py --file todo.txt --dialogue
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from mood import MoodAnalyzer

SHOWCASE_SAMPLES = [
    {
        "mood": "happy_excited",
        "text": "I can't believe we actually won! This is the most amazing feeling ever!!",
        "lang": "en",
    },
    {
        "mood": "exhausted_sad",
        "text": "I am so tired... I worked all night and I just want to sleep...",
        "lang": "en",
    },
    {
        "mood": "angry",
        "text": "Stop it right now! I told you not to touch that!!",
        "lang": "en",
    },
    {
        "mood": "japanese_exhausted",
        "text": "最近、仕事が忙しすぎて、寝る暇もないよ……。",
        "lang": "ja",
    },
    {
        "mood": "japanese_tender",
        "text": "大丈夫だよ。一人で抱え込まずに、頼れる人には頼るべきだと思うよ。",
        "lang": "ja",
    },
]


def synthesize_with_mood(
    model,
    text: str,
    language: str,
    ref_wav: Path,
    out_path: Path,
    engine_name: str = "xtts",
):
    analyzer = MoodAnalyzer()
    mood_result = analyzer.analyze(text, language=language)

    print(f"\n[Mood Detected] {mood_result.primary_emotion.upper()} "
          f"(intensity: {mood_result.intensity:.2f}, valence: {mood_result.valence:+.2f})")
    print(f"  Speed modifier : {mood_result.prosody.speed_ratio:.2f}x")
    print(f"  Pitch semitones: {mood_result.prosody.pitch_shift_semitones:+.1f}st")
    print(f"  Breathiness    : {mood_result.prosody.breathiness:.2f}")
    print(f"  Fish Audio tag : {mood_result.format_for_fish_speech()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine_name == "fish-speech":
        from engines.registry import get_engine
        engine = get_engine("fish-speech")
        tagged_text = mood_result.format_for_fish_speech()
        return engine.synthesize(
            text=tagged_text,
            out=out_path,
            language=language,
            ref_audio=ref_wav,
        )

    # XTTS synthesis
    from TTS.api import TTS
    xtts = model.synthesizer.tts_model

    # Check for precomputed latents cache (.pt)
    pt_path = ref_wav.with_suffix(".pt")
    if pt_path.exists():
        lat = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    else:
        gpt, spk = xtts.get_conditioning_latents(audio_path=str(ref_wav))
        lat = {"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}

    # Map mood to XTTS synthesis parameters:
    # High intensity / excited -> higher temperature & faster speed
    # Sad / exhausted -> lower temperature & slower speed
    base_temp = 0.70
    if mood_result.primary_emotion in ["excited", "joy", "anger"]:
        temperature = min(0.95, base_temp + (mood_result.intensity * 0.20))
    elif mood_result.primary_emotion in ["exhaustion", "tired", "sadness", "whisper"]:
        temperature = max(0.55, base_temp - (mood_result.intensity * 0.15))
    else:
        temperature = base_temp

    speed = mood_result.prosody.speed_ratio

    print(f"  -> Synthesizing (speed={speed:.2f}, temp={temperature:.2f})...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = xtts.inference(
            text=text,
            language=language,
            gpt_cond_latent=lat["gpt_cond_latent"],
            speaker_embedding=lat["speaker_embedding"],
            temperature=temperature,
            speed=speed,
            enable_text_splitting=True,
        )["wav"]

    model.synthesizer.save_wav(wav, str(out_path))
    dur = time.perf_counter() - t0
    print(f"  ✔ Generated in {dur:.2f}s -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Test speech sound with emotional mood modulation.")
    parser.add_argument("--text", "-t", help="Custom text to synthesize")
    parser.add_argument("--language", "-l", default="auto", help="Language code (en, ja, or auto)")
    parser.add_argument("--ref", "-r", default="data/audio_clean/deer_expressive.wav", help="Reference voice audio")
    parser.add_argument("--engine", default="xtts", choices=["xtts", "fish-speech"], help="TTS engine")
    parser.add_argument("--preset", choices=["showcase"], help="Run preset showcase across emotions")
    parser.add_argument("--out-dir", default="outputs/mood_test", help="Output directory")

    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        # Fallback to any available clean ref wav
        candidates = list(Path("data/audio_clean").glob("*.wav"))
        if candidates:
            ref_path = candidates[0]
            print(f"Note: Using reference voice: {ref_path}")
        else:
            print(f"Error: Reference voice not found: {args.ref}", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model if XTTS
    model = None
    if args.engine == "xtts":
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading XTTS model on {device}...")
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    if args.preset == "showcase":
        print("=== Running Mood Voice Showcase ===")
        for item in SHOWCASE_SAMPLES:
            out_file = out_dir / f"showcase_{item['mood']}.wav"
            print(f"\n--- Testing Mood: {item['mood']} ---")
            print(f"Text: \"{item['text']}\"")
            synthesize_with_mood(
                model=model,
                text=item["text"],
                language=item["lang"],
                ref_wav=ref_path,
                out_path=out_file,
                engine_name=args.engine,
            )
        print(f"\nAll showcase audio files saved in: {out_dir.resolve()}")
    elif args.text:
        lang = args.language
        if lang == "auto":
            from mood.ml_classifier import MLMoodClassifier
            lang = MLMoodClassifier.detect_language(args.text)

        out_file = out_dir / "mood_test.wav"
        synthesize_with_mood(
            model=model,
            text=args.text,
            language=lang,
            ref_wav=ref_path,
            out_path=out_file,
            engine_name=args.engine,
        )
        print(f"\nAudio ready! You can listen at: {out_file.resolve()}")
    else:
        print("Please provide --text or --preset showcase. Run with --help for details.")


if __name__ == "__main__":
    main()
