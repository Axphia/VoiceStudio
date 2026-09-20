"""Calibrate voice output: pitch, expressiveness, speed, epoch pick.

  uv run python scripts/calibrate.py --text "สวัสดีครับ" --voice data/audio_clean/deer_expressive.wav --pitch 2 --temperature 0.85 --out outputs/cali.wav

  --pitch N       semitones, -12..+12 (2 = slightly brighter, -2 = deeper)
  --temperature   0.6 flat/safe, 0.75 default, 0.85 expressive
  --model DIR     exported bundle from export_model.py (= epoch pick).
                  Default: base XTTS (no fine-tune).
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import librosa
import soundfile as sf
import torch

OUT_SR = 24000  # XTTS native output rate


def pitch_shift_resample(y, n_steps: float):
    """Pitch shift preserving duration (librosa's shifter is broken here:
    numba/numpy ufunc conflict). Two resamples = same effect, tiny quality cost."""
    import numpy as np
    from scipy.signal import resample

    rate = 2.0 ** (n_steps / 12.0)
    shifted = resample(y, max(1, int(len(y) / rate)))
    return np.asarray(resample(shifted, len(y)), dtype=np.float32)


def load_model(model_dir: str | None):
    from TTS.api import TTS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if model_dir:
        # NB: TTS model_path is the bundle DIRECTORY (it appends model.pth).
        b = Path(model_dir)
        model = TTS(model_path=str(b),
                    config_path=str(b / "config.json")).to(dev)
    else:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    print(f"[calibrate] model on {dev}: {model_dir or 'base xtts-v2'}")
    return model


def main(text, voice, out, language, model_dir, pitch, temperature, speed):
    model = load_model(model_dir)
    lat = model.synthesizer.tts_model.get_conditioning_latents(
        audio_path=str(voice))
    if isinstance(lat, tuple):
        gpt_lat, spk_emb = lat
    else:
        gpt_lat, spk_emb = lat["gpt_cond_latent"], lat["speaker_embedding"]
    dev = next(model.synthesizer.tts_model.parameters()).device
    with torch.inference_mode():
        wav = model.synthesizer.tts_model.inference(
            text.strip(), language, gpt_lat.to(dev), spk_emb.to(dev),
            temperature=temperature, speed=speed,
            enable_text_splitting=True)["wav"]
    import numpy as np

    y = np.asarray(wav, dtype=np.float32)
    if abs(pitch) > 0.01:
        y = pitch_shift_resample(y, pitch)
        print(f"[calibrate] pitch {pitch:+g} semitones")
    print(f"[calibrate] temperature {temperature}, speed {speed}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, OUT_SR)
    print(f"[calibrate] Saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--out", default="outputs/calibrated.wav")
    p.add_argument("--language", default="ja")
    p.add_argument("--model", default=None, help="export bundle dir (epoch pick)")
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()
    main(a.text, Path(a.voice), Path(a.out), a.language, a.model,
         a.pitch, a.temperature, a.speed)
