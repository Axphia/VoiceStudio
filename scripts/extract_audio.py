"""Extract clean WAV audio from a video file for zero-shot voice cloning."""
import argparse
import subprocess
from pathlib import Path


def extract_audio(video: Path, out: Path, sr: int = 22050):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",               # no video
        "-ac", "1",          # mono (XTTS prefers mono)
        "-ar", str(sr),      # sample rate
        "-c:a", "pcm_s16le",
        str(out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video, e.g. data/videos/input.mp4")
    p.add_argument("--out", default="data/audio_clean/ref.wav")
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()
    extract_audio(Path(args.video), Path(args.out), args.sr)
