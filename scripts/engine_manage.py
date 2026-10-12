"""Manage TTS engines.

  uv run python scripts/engines.py list
  uv run python scripts/engines.py add git+https://github.com/user/their-engine.git
  uv run python scripts/engines.py test --engine xtts --text "hello" --language en
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from engines.registry import get_engine, list_engines


def cmd_list(_args):
    for name, cls in sorted(list_engines().items()):
        inst = cls()
        print(f"- {name}: {inst.label} (needs_ref={inst.needs_ref})")


def cmd_add(args):
    # Installs a third-party engine package straight from GitHub.
    # The package must expose a VoiceEngine subclass via the
    # "voicestudio.engines" entry-point group to auto-register.
    cmd = [sys.executable, "-m", "pip", "install", args.url]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Installed. Engines now:")
    cmd_list(args)


def cmd_test(args):
    engine = get_engine(args.engine)
    ref = Path(args.ref) if args.ref else None
    out = engine.synthesize(args.text, Path(args.out), args.language,
                            ref_audio=ref, speed=args.speed)
    print(f"OK -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("add")
    s.add_argument("url", help="e.g. git+https://github.com/user/engine.git")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("test")
    s.add_argument("--engine", default="xtts")
    s.add_argument("--text", default="Hello from VoiceStudio.")
    s.add_argument("--ref", default="data/audio_clean/deer_expressive.wav")
    s.add_argument("--out", default="outputs/engine_test.wav")
    s.add_argument("--language", default="en")
    s.add_argument("--speed", type=float, default=1.0)
    s.set_defaults(fn=cmd_test)

    args = p.parse_args()
    args.fn(args)
