"""Hardware profiler: detect CPU/RAM/VRAM and pick lag-free settings.

  uv run python scripts/hw.py              # print report + recommended profile
  uv run python scripts/hw.py --apply mid  # print effective train settings

Profiles (train_xtts.py --profile auto|low|mid|high):
  low  - weak iGPU / <=4GB VRAM / <=8GB RAM: batch 1, accum 8, workers 0, fp32
  mid  - RTX 3050 4GB class: batch 1, accum 16, workers 0, fp32, capped threads
  high - 10GB+ VRAM: batch 2, accum 8, workers 2, fp16 allowed

Why this stops lag: workers=0 kills dataloader subprocesses, capped torch
threads leave CPU for the desktop, and batch 1 + accumulation keeps VRAM
headroom so Windows stays responsive.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import torch

PROFILES = {
    "low": {"batch_size": 1, "grad_accum": 8, "num_workers": 0,
            "fp16": False, "torch_threads": 2,
            "note": "minimum footprint; slowest training"},
    "mid": {"batch_size": 1, "grad_accum": 16, "num_workers": 0,
            "fp16": False, "torch_threads": 4,
            "note": "balanced: full speed the GPU allows, desktop stays usable"},
    "high": {"batch_size": 2, "grad_accum": 8, "num_workers": 2,
             "fp16": True, "torch_threads": 8,
             "note": "fast; needs 10GB+ VRAM"},
}


def detect() -> dict:
    vm = psutil.virtual_memory()
    info = {
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_avail_gb": round(vm.available / 1e9, 1),
        "cuda": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_total_gb": 0.0,
    }
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(p.total_memory / 1e9, 2)
        try:
            free, _ = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = round(free / 1e9, 2)
        except Exception:
            info["vram_free_gb"] = -1.0
    return info


def recommend(hw: dict) -> str:
    if not hw["cuda"] or hw["ram_total_gb"] <= 8:
        return "low"
    if hw["vram_total_gb"] >= 10:
        return "high"
    return "mid"  # e.g. RTX 3050 4GB: batch 1 + workers 0 keeps it usable


def apply(profile: str) -> dict:
    """Cap process-wide CPU usage; return the train settings dict."""
    s = PROFILES[profile]
    torch.set_num_threads(max(1, min(s["torch_threads"],
                                     psutil.cpu_count(logical=True) or 4)))
    torch.set_num_interop_threads(1)
    return s


def report(profile: str | None = None):
    hw = detect()
    rec = recommend(hw)
    use = profile or rec
    print("--- hardware ---")
    print(f"CPU : {hw['cpu_physical']} cores / {hw['cpu_logical']} threads")
    print(f"RAM : {hw['ram_avail_gb']}/{hw['ram_total_gb']} GB free")
    if hw["cuda"]:
        print(f"GPU : {hw['gpu_name']} ({hw['vram_total_gb']} GB VRAM)")
        if hw.get("vram_free_gb", -1) >= 0:
            print(f"VRAM free: {hw['vram_free_gb']} GB", end="")
            if hw["vram_free_gb"] < 3.5:
                print("  <-- LOW! close browser/games before training")
            else:
                print()
    else:
        print("GPU : none (CPU only)")
    print(f"profile: {use} (auto={rec}) — {PROFILES[use]['note']}")
    print("--- effective settings ---")
    for k, v in PROFILES[use].items():
        if k != "note":
            print(f"{k}: {v}")
    return use


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", default=None, choices=["low", "mid", "high"],
                   help="force a profile instead of auto")
    a = p.parse_args()
    report(a.apply)
