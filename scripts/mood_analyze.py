#!/usr/bin/env python3
"""CLI utility to analyze text mood, emotion, and prosody parameters."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mood import MoodAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze mood, emotion, and prosody from text.")
    parser.add_argument("text", nargs="?", default=None, help="Text to analyze (or use --file)")
    parser.add_argument("--file", "-f", type=Path, help="Path to text or script file to analyze")
    parser.add_argument("--mode", choices=["auto", "ml", "llm"], default="auto", help="Analysis mode")
    parser.add_argument("--dialogue", "-d", action="store_true", help="Parse as multi-speaker dialogue script")
    parser.add_argument("--engine", choices=["fish-speech", "bert-vits2"], default="fish-speech", help="Target TTS engine format")
    parser.add_argument("--json", action="store_true", help="Output full analysis as JSON")
    parser.add_argument("--llm-url", help="Base URL for OpenAI-compatible LLM endpoint")
    parser.add_argument("--llm-model", help="LLM model name (e.g., qwen2.5:7b)")

    args = parser.parse_args()

    content = ""
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        content = args.file.read_text(encoding="utf-8")
    elif args.text:
        content = args.text
    else:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    analyzer = MoodAnalyzer(
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
    )

    if args.dialogue:
        results = analyzer.analyze_dialogue(content, mode=args.mode)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            print(f"\n--- Dialogue Mood & Prosody Analysis ({len(results)} lines) ---")
            for i, r in enumerate(results, 1):
                speaker_prefix = f"[{r.speaker}] " if r.speaker else ""
                print(f"\n#{i}: {speaker_prefix}{r.text}")
                print(f"   Emotion: {r.primary_emotion.upper()} (intensity: {r.intensity:.2f}, confidence: {r.confidence:.2f})")
                print(f"   Prosody: pitch: {r.prosody.pitch_shift_semitones:+.1f}st | speed: {r.prosody.speed_ratio:.2f}x | energy: {r.prosody.energy_ratio:.2f}x | breath: {r.prosody.breathiness:.2f}")
                print(f"   TTS Tagged: {r.format_for_fish_speech()}")
    else:
        result = analyzer.analyze(content, mode=args.mode)
        if args.json:
            print(result.to_json())
        else:
            print(f"\n--- Mood & Prosody Analysis ---")
            print(f"Text:            {result.text}")
            print(f"Primary Emotion: {result.primary_emotion.upper()} (confidence: {result.confidence:.2f})")
            print(f"Intensity:       {result.intensity:.2f} / 1.00")
            print(f"Valence/Arousal: {result.valence:+.2f} / {result.arousal:.2f}")
            print(f"Prosody (SSML):  <prosody {result.prosody.to_ssml_attributes()}>")
            print(f"Fish Speech:     {result.format_for_fish_speech()}")
            print(f"Top Emotions:    {', '.join(f'{e.emotion}: {e.score:.2f}' for e in result.emotions[:3])}")


if __name__ == "__main__":
    main()
