"""Export a trained XTTS checkpoint to an inference-ready .pth bundle.

Trainer checkpoints (checkpoint_N.pth, one per epoch -> pick your epoch) hold
the full model. This repacks one into the folder layout the TTS API loads:

  models/<name>/model.pth  config.json  vocab.json  speakers_xtts.pth

Usage:
  uv run python scripts/export_model.py --checkpoint training/run/<run>/checkpoint_9.pth --out models/myvoice
  uv run python scripts/export_model.py --from-base --out models/base_copy   # test the pipeline, no training needed

Notes:
- .pth bundle works with this repo, RVC-style changers do NOT read XTTS
  weights (different architecture). Full XTTS->ONNX is not practical
  (autoregressive GPT); the bundle below is the portable artifact.
- Epoch calibration = choose which checkpoint_N.pth to export.
"""
import argparse
import os
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import torch

BASE_DIR = (Path(os.environ.get("LOCALAPPDATA", "~"))
            / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2")


def load_state(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    return ckpt.get("model", ckpt)


def export(checkpoint: Path, out: Path, base_dir: Path):
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(checkpoint)
    prefs = sorted({k.split(".")[0] for k in state})
    print(f"[export] weight groups: {prefs}")
    if "gpt" not in prefs or "hifigan_decoder" not in prefs:
        raise SystemExit(f"Not an XTTS checkpoint: {checkpoint}")

    torch.save({"model": state}, str(out / "model.pth"))
    run_cfg = checkpoint.parent / "config.json"
    shutil.copy(run_cfg if run_cfg.exists() else base_dir / "config.json",
                out / "config.json")
    for f in ("vocab.json", "speakers_xtts.pth"):
        shutil.copy(base_dir / f, out / f)
    print(f"[export] bundle -> {out} "
          f"({sum(p.stat().st_size for p in out.iterdir())/1e9:.2f} GB)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="trainer checkpoint_N.pth (epoch pick)")
    p.add_argument("--from-base", action="store_true",
                   help="export the downloaded base model (pipeline test)")
    p.add_argument("--out", required=True, help="e.g. models/myvoice")
    p.add_argument("--base-dir", default=str(BASE_DIR))
    a = p.parse_args()
    if a.checkpoint:
        ckpt = Path(a.checkpoint)
    elif a.from_base:
        ckpt = Path(a.base_dir).expanduser() / "model.pth"
    else:
        p.error("give --checkpoint or --from-base")
    export(ckpt, Path(a.out), Path(a.base_dir).expanduser())
