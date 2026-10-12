"""Compute XTTS speaker latents once and cache to .pt for fast reuse.

Biggest win on 4GB GPUs: get_conditioning_latents (audio encode + GPT
conditioning) runs once instead of on every clone call.

Usage:
  uv run python scripts/compute_latents.py --speaker-wav data/audio_clean/deer_expressive.wav --out data/audio_clean/deer.pt
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch
from TTS.api import TTS


def main(speaker_wav: Path, out: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t0 = time.perf_counter()
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Model load: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    xtts = model.synthesizer.tts_model
    gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
        audio_path=str(speaker_wav),
    )
    print(f"Latents: {time.perf_counter() - t0:.1f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
        str(out),
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speaker-wav", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(Path(args.speaker_wav), Path(args.out))
