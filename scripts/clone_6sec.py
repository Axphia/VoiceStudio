#!/usr/bin/env python3
"""Clone voice using exactly a 6-second clean audio sample.

XTTS works best with 6 to 10 seconds of clean reference speech.
This script:
  1. Trims and cleans exactly 6.0 seconds from any audio or video file.
  2. Applies noise reduction and level normalization.
  3. Caches .pt latents for fast repeat clones.
  4. Synthesizes cloned speech (optionally with mood auto-prosody).

Usage examples:
  # Just prepare a 6-second clean voice reference from a file or video:
  python scripts/clone_6sec.py --in "data/audio_clean/deer_ref.wav" --name "my_voice_6s"

  # Clean 6s AND immediately speak new text:
  python scripts/clone_6sec.py --in "data/audio_clean/deer_ref.wav" --text "Hello, this is my cloned voice!" --language en

  # Clone with Japanese text + auto mood:
  python scripts/clone_6sec.py --in "data/audio_clean/deer_ref.wav" --text "最近、仕事が忙しすぎて、寝る暇もないよ。" --language ja --mood
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch


def clean_to_6sec(
    inp_path: Path,
    out_wav: Path,
    start_sec: float = 0.0,
    target_dur: float = 6.0,
    sr: int = 22050,
    denoise_strength: float = 0.4,
) -> Path:
    """Loads audio, selects optimal 6 seconds, denoises, and normalizes."""
    print(f"\n[1/3] Extracting 6-second clean reference from: {inp_path.name}")
    
    # Load audio (works with wav, mp3, mp4, etc. via ffmpeg/librosa)
    y, orig_sr = librosa.load(str(inp_path), sr=sr, mono=True)
    total_sec = len(y) / sr
    print(f"      Source duration: {total_sec:.2f}s")

    # Trim initial silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    s_idx = int(start_sec * sr)
    e_idx = s_idx + int(target_dur * sr)

    if len(y_trimmed) >= int(target_dur * sr):
        segment = y_trimmed[s_idx:e_idx]
    else:
        # If source is shorter than 6s, pad gently
        print(f"      Warning: Source ({len(y_trimmed)/sr:.2f}s) is shorter than {target_dur}s. Using full available audio.")
        segment = y_trimmed

    # Denoise (0.3 - 0.5 preserves natural vocal warmth)
    if denoise_strength > 0:
        segment = nr.reduce_noise(y=segment, sr=sr, prop_decrease=denoise_strength, stationary=True)

    # Normalize to -3 dBFS peak
    peak = np.abs(segment).max() + 1e-9
    segment = (segment / peak) * 0.70

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), segment, sr)
    actual_dur = len(segment) / sr
    print(f"      ✔ Clean reference saved: {out_wav} ({actual_dur:.2f} seconds)")
    return out_wav


def compute_latents(model, clean_wav: Path, out_pt: Path) -> dict:
    """Computes and caches XTTS conditioning latents."""
    print(f"\n[2/3] Computing speaker latents...")
    t0 = time.perf_counter()
    xtts = model.synthesizer.tts_model
    gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(audio_path=str(clean_wav))
    lat = {"gpt_cond_latent": gpt_cond_latent.cpu(), "speaker_embedding": speaker_embedding.cpu()}
    torch.save(lat, str(out_pt))
    print(f"      ✔ Latents cached in {time.perf_counter() - t0:.2f}s -> {out_pt.name}")
    return lat


def clone_speech(
    model,
    text: str,
    language: str,
    latents: dict,
    out_wav: Path,
    speed: float = 1.0,
    temperature: float = 0.75,
    enable_mood: bool = False,
):
    """Synthesizes text using conditioned latents."""
    print(f"\n[3/3] Synthesizing cloned speech...")
    
    if enable_mood:
        from mood import MoodAnalyzer
        analyzer = MoodAnalyzer()
        res = analyzer.analyze(text, language=language)
        print(f"      Mood: {res.primary_emotion.upper()} (intensity: {res.intensity:.2f})")
        speed = res.prosody.speed_ratio
        if res.primary_emotion in ["excited", "joy", "anger"]:
            temperature = min(0.95, 0.70 + (res.intensity * 0.20))
        elif res.primary_emotion in ["exhaustion", "tired", "sadness", "whisper"]:
            temperature = max(0.55, 0.70 - (res.intensity * 0.15))
        print(f"      Applied: speed={speed:.2f}x, temp={temperature:.2f}")

    t0 = time.perf_counter()
    xtts = model.synthesizer.tts_model
    with torch.inference_mode():
        wav = xtts.inference(
            text=text,
            language=language,
            gpt_cond_latent=latents["gpt_cond_latent"],
            speaker_embedding=latents["speaker_embedding"],
            temperature=temperature,
            speed=speed,
            enable_text_splitting=True,
        )["wav"]

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    model.synthesizer.save_wav(wav, str(out_wav))
    print(f"      ✔ Done in {time.perf_counter() - t0:.2f}s -> {out_wav}")


def main():
    parser = argparse.ArgumentParser(description="Voice cloning using 6 seconds of clean reference sound.")
    parser.add_argument("--in", "-i", dest="inp", required=True, help="Input audio or video file")
    parser.add_argument("--name", "-n", default="cloned_6s", help="Name identifier for the clean reference")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds inside input file")
    parser.add_argument("--dur", type=float, default=6.0, help="Duration in seconds to extract (default: 6.0)")
    parser.add_argument("--text", "-t", default=None, help="Text to synthesize with the 6s voice")
    parser.add_argument("--language", "-l", default="en", help="Language code (en, ja, etc.)")
    parser.add_argument("--out", "-o", default=None, help="Output wav path for synthesized speech")
    parser.add_argument("--mood", action="store_true", help="Auto-adjust speed/tone based on text emotion")
    parser.add_argument("--denoise", type=float, default=0.4, help="Denoising strength (0.0 to 0.8)")

    args = parser.parse_args()

    inp_file = Path(args.inp)
    if not inp_file.exists():
        print(f"Error: Input file does not exist: {inp_file}", file=sys.stderr)
        sys.exit(1)

    clean_dir = ROOT / "data" / "audio_clean"
    clean_wav = clean_dir / f"{args.name}_clean.wav"
    clean_pt = clean_dir / f"{args.name}_clean.pt"

    # Step 1: Extract 6s clean sound
    clean_to_6sec(
        inp_path=inp_file,
        out_wav=clean_wav,
        start_sec=args.start,
        target_dur=args.dur,
        denoise_strength=args.denoise,
    )

    # Step 2: Compute latents and optionally clone speech
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS model on {device}...")
    from TTS.api import TTS
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    latents = compute_latents(model, clean_wav, clean_pt)

    # Step 3: Synthesize if text was supplied
    if args.text:
        out_path = Path(args.out) if args.out else ROOT / "outputs" / f"{args.name}_speech.wav"
        clone_speech(
            model=model,
            text=args.text,
            language=args.language,
            latents=latents,
            out_wav=out_path,
            enable_mood=args.mood,
        )
        print(f"\n✨ Cloned speech audio is ready at:\n   {out_path.resolve()}")
    else:
        print(f"\n✨ Clean 6-second reference audio and .pt latents ready in data/audio_clean/!")
        print(f"   Audio:   {clean_wav.resolve()}")
        print(f"   Latents: {clean_pt.resolve()}")
        print(f"\nYou can now clone any speech with:")
        print(f"   python scripts/clone_fast.py --latents \"{clean_pt}\" --text \"Your text here\" --language {args.language}")


if __name__ == "__main__":
    main()
