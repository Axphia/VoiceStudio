"""Clean a reference WAV: trim, denoise, normalize for XTTS."""
import argparse
from pathlib import Path

import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np


def clean(inp: Path, out: Path, start: float = 0.0, dur: float = 10.0, sr: int = 22050):
    y, _ = librosa.load(str(inp), sr=sr, mono=True)
    s = int(start * sr)
    e = int(min(len(y), (start + dur) * sr))
    y = y[s:e]
    # denoise (stationary, small prop to avoid artifacts)
    y_dn = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6, stationary=True)
    # normalize to -3 dBFS peak, trim leading/trailing silence
    y_dn, _ = librosa.effects.trim(y_dn, top_db=30)
    peak = np.abs(y_dn).max() + 1e-9
    y_dn = y_dn / peak * 0.7
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y_dn, sr)
    print(f"Saved {len(y_dn)/sr:.1f}s -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=3.0)
    p.add_argument("--dur", type=float, default=8.0)
    args = p.parse_args()
    clean(Path(args.inp), Path(args.out), args.start, args.dur)
