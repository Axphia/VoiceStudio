"""Fast dataset preparation using the compiled Rust voice-prep engine.

Usage:
  uv run python scripts/prepare_fast.py --input data/audio_clean --out data/ft/myvoice --sr 22050
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST_BIN = ROOT / "crates" / "voice_core" / "target" / "release" / "voice-prep.exe"


def main():
    p = argparse.ArgumentParser(description="Fast Rust-powered Voice Dataset Preparation")
    p.add_argument("--input", required=True, help="Input directory containing WAV files")
    p.add_argument("--out", required=True, help="Output dataset directory (e.g. data/ft/myvoice)")
    p.add_argument("--sr", type=int, default=22050, help="Target sample rate (default 22050 for XTTS)")
    p.add_argument("--max-dur", type=float, default=11.0, help="Maximum segment duration in seconds")
    p.add_argument("--min-dur", type=float, default=1.0, help="Minimum segment duration in seconds")
    p.add_argument("--peak", type=float, default=0.707, help="Peak normalization target (-3 dBFS)")
    p.add_argument("--top-db", type=float, default=30.0, help="Silence trim threshold in dB")
    args = p.parse_args()

    if not RUST_BIN.exists():
        print(f"[prepare_fast] Rust binary not found at {RUST_BIN}. Compiling with cargo...")
        cmd_build = ["cargo", "build", "--release"]
        res = subprocess.run(cmd_build, cwd=ROOT / "crates" / "voice_core")
        if res.returncode != 0:
            sys.exit("[prepare_fast] Failed to build Rust preprocessor binary.")

    cmd = [
        str(RUST_BIN),
        "batch-prep",
        "--input-dir", str(Path(args.input).resolve()),
        "--out-dir", str(Path(args.out).resolve()),
        "--sr", str(args.sr),
        "--max-dur", str(args.max_dur),
        "--min-dur", str(args.min_dur),
        "--peak", str(args.peak),
        "--top-db", str(args.top_db),
    ]

    print(f"[prepare_fast] Running Rust engine: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(f"[prepare_fast] Error during batch prep (exit code {res.returncode})")
    print(f"[prepare_fast] Finished successfully! Dataset ready at: {args.out}")


if __name__ == "__main__":
    main()
