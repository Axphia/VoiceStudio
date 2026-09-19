"""Zero-shot voice cloning with Coqui XTTS-v2. No training needed."""
import argparse
from pathlib import Path

import torch
from TTS.api import TTS


def main(text: str, speaker_wav: Path, out: Path, language: str = "en", speed: float = 1.0, split: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(out),
        speed=speed,
        split_sentences=split,
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--speaker-wav", default="data/audio_clean/ref.wav")
    p.add_argument("--out", default="outputs/cloned.wav")
    p.add_argument("--language", default="en")
    p.add_argument("--speed", type=float, default=1.0, help="0.9 slower/more natural, 1.0 normal")
    p.add_argument("--no-split", action="store_true", help="disable sentence splitting (flatter)")
    args = p.parse_args()
    main(args.text, Path(args.speaker_wav), Path(args.out), args.language, args.speed, not args.no_split)
