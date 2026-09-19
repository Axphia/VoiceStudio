"""Fast batch cloning: model loads ONCE, latents load from cache.

  uv run python scripts/compute_latents.py --speaker-wav data/audio_clean/deer_expressive.wav --out data/audio_clean/deer.pt
  uv run python scripts/clone_fast.py --latents data/audio_clean/deer.pt --texts texts.txt --out-dir outputs/batch --language ja

Single text also works:
  uv run python scripts/clone_fast.py --latents data/audio_clean/deer.pt --text "こんにちは！" --out outputs/quick.wav --language ja
"""
import argparse
import time
import warnings
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch
from TTS.api import TTS


def optimize_for_3050():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t0 = time.perf_counter()
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Model load: {time.perf_counter() - t0:.1f}s")
    return model


def synth(model, text, language, latents, out, temperature, speed):
    xtts = model.synthesizer.tts_model
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = xtts.inference(
            text,
            language,
            latents["gpt_cond_latent"],
            latents["speaker_embedding"],
            temperature=temperature,
            speed=speed,
            enable_text_splitting=True,
        )["wav"]
    out.parent.mkdir(parents=True, exist_ok=True)
    model.synthesizer.save_wav(wav, str(out))
    print(f"[{time.perf_counter() - t0:.1f}s] Saved -> {out}")


def main(latents_path, text, texts, out, out_dir, language, temperature, speed,
         engine="xtts", ref_audio=None):
    optimize_for_3050()
    if engine != "xtts":
        from engines.registry import get_engine

        eng = get_engine(engine)
        jobs = []
        if texts:
            lines = [l.strip() for l in Path(texts).read_text(encoding="utf-8").splitlines() if l.strip()]
            jobs = [(line, Path(out_dir) / f"line_{i:02d}.wav") for i, line in enumerate(lines)]
        else:
            jobs = [(text, Path(out))]
        for line, dest in jobs:
            eng.synthesize(line, dest, language,
                           ref_audio=Path(ref_audio) if ref_audio else None,
                           speed=speed)
        return
    model = load_model()
    latents = torch.load(str(latents_path), map_location="cpu", weights_only=False)
    # move conditioning tensors to model device
    dev = next(model.synthesizer.tts_model.parameters()).device
    latents = {
        "gpt_cond_latent": latents["gpt_cond_latent"].to(dev),
        "speaker_embedding": latents["speaker_embedding"].to(dev),
    }

    if texts:  # batch file, one sentence per line
        lines = [l.strip() for l in Path(texts).read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, line in enumerate(lines):
            synth(model, line, language, latents, Path(out_dir) / f"line_{i:02d}.wav", temperature, speed)
    else:
        synth(model, text, language, latents, Path(out), temperature, speed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default=None, help=".pt from compute_latents.py (xtts fast path)")
    p.add_argument("--text", default=None)
    p.add_argument("--texts", default=None, help="text file, one sentence per line (batch)")
    p.add_argument("--out", default="outputs/quick.wav")
    p.add_argument("--out-dir", default="outputs/batch")
    p.add_argument("--language", default="ja")
    p.add_argument("--temperature", type=float, default=0.75,
                   help="0.6 flatter/safer, 0.75 default, 0.85 more expressive")
    p.add_argument("--speed", type=float, default=0.95)
    p.add_argument("--engine", default="xtts", help="xtts (fast path) or a registered external engine")
    p.add_argument("--ref-audio", default=None, help="reference WAV for non-xtts engines")
    ARGS = p.parse_args()
    if not ARGS.text and not ARGS.texts:
        p.error("give --text or --texts")
    if ARGS.engine == "xtts" and not ARGS.latents:
        p.error("--latents is required for the xtts fast path")
    main(Path(ARGS.latents) if ARGS.latents else None, ARGS.text, ARGS.texts, ARGS.out, ARGS.out_dir,
         ARGS.language, ARGS.temperature, ARGS.speed, ARGS.engine, ARGS.ref_audio)
