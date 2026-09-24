"""Fine-tune XTTS-v2 on YOUR voice (real training, not zero-shot).

Dataset first:
  uv run python scripts/prepare_ft.py --input data/audio_clean --out data/ft/myvoice --language ja

Smoke test (overfits 1 batch — proves the graph fits your 4GB VRAM):
  uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --smoke

Full run (slow on RTX 3050 4GB; batch_size=1 + grad accumulation + fp16):
  uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --epochs 20

Resume:
  uv run python scripts/train_xtts.py --dataset data/ft/myvoice --language ja --resume training/run/XTTS_FT-*.pth
"""
import argparse
import dataclasses
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig as TrainerAudioConfig,
)

import platform
if platform.system() == "Windows":
    _tts_home = Path(os.environ.get("LOCALAPPDATA", "~")) / "tts"
else:
    _tts_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tts"

BASE_DIR = _tts_home / "tts_models--multilingual--multi-dataset--xtts_v2"


def compute_mel_stats(dataset: Path, out_path: Path):
    """Fallback per-mel-bin std when the base checkpoint has no mel_stats."""
    from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

    spec = TorchMelSpectrogram(sampling_rate=22050, normalize=False)
    tot = torch.zeros(80, dtype=torch.float64)
    tot2 = torch.zeros(80, dtype=torch.float64)
    n = 0
    wavs = sorted((dataset / "wavs").glob("*.wav"))
    import soundfile as sf

    with torch.inference_mode():
        for w in wavs:
            y, sr = sf.read(str(w))
            assert sr == 22050, f"{w} is {sr}Hz, need 22050"
            m = spec(torch.tensor(y, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            tot += m.sum(dim=1).double()
            tot2 += (m ** 2).sum(dim=1).double()
            n += m.shape[1]
    mean = tot / n
    std = ((tot2 / n - mean ** 2).clamp_min(1e-6)).sqrt().float()
    torch.save(std, str(out_path))
    print(f"[train] mel stats ({len(wavs)} files) -> {out_path}")
    return out_path


def prepare_base_assets(model_dir: Path, dataset: Path, out: Path,
                        dvae_arg: str | None):
    """Official mel_stats (from base checkpoint) + DVAE (from HuggingFace).

    The Coqui gateway URLs baked into TTS are dead; HF coqui/XTTS-v2 works.
    """
    ckpt = torch.load(str(model_dir / "model.pth"), map_location="cpu")
    state = ckpt.get("model", ckpt)

    mel_path = out / "mel_stats.pth"
    if not mel_path.exists():
        if "mel_stats" in state:
            torch.save(state["mel_stats"], str(mel_path))
            print(f"[train] official mel_stats extracted -> {mel_path}")
        else:
            compute_mel_stats(dataset, mel_path)

    if dvae_arg:
        dvae_path = Path(dvae_arg)
    else:
        dvae_path = out / "dvae.pth"
        if not dvae_path.exists():
            from huggingface_hub import hf_hub_download

            dl = hf_hub_download("coqui/XTTS-v2", "dvae.pth")
            import shutil

            shutil.copy(dl, dvae_path)
            print(f"[train] DVAE downloaded -> {dvae_path}")
    return mel_path, dvae_path


def train(dataset, language, out, epochs, eval_split, smoke, resume,
          base_dir, test_text, mel_stats, dvae_ckpt, fp16, grad_accum,
          save_step, profile, run_name, burst_epochs=0, rest_seconds=90):
    model_dir = Path(base_dir).expanduser()
    for f in ("config.json", "model.pth", "vocab.json"):
        if not (model_dir / f).exists():
            raise SystemExit(f"Base model file missing: {model_dir / f} "
                             f"(run clone.py once to download XTTS-v2)")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if mel_stats:
        mel_path = Path(mel_stats)
        _, dvae_path = prepare_base_assets(
            model_dir, Path(dataset), out, dvae_ckpt)
    else:
        mel_path, dvae_path = prepare_base_assets(
            model_dir, Path(dataset), out, dvae_ckpt)

    config = GPTTrainerConfig()
    config.load_json(str(model_dir / "config.json"))

    # Base config.json carries inference-only sections; merge into trainer types.
    base_audio = config.audio
    config.audio = TrainerAudioConfig(**{
        f.name: getattr(base_audio, f.name)
        for f in dataclasses.fields(TrainerAudioConfig)
        if hasattr(base_audio, f.name)})

    # --- fine-tune identity ---
    # Base config.json carries inference-only XttsArgs (no mel_norm_file etc.),
    # so merge its architecture values into a trainer GPTArgs (keeps 1026 tokens).
    base_args = config.model_args
    vals = {f.name: getattr(base_args, f.name)
            for f in dataclasses.fields(GPTArgs) if hasattr(base_args, f.name)}
    model_args = GPTArgs(**vals)
    model_args.xtts_checkpoint = str(model_dir / "model.pth")
    model_args.tokenizer_file = str(model_dir / "vocab.json")
    model_args.max_text_length = 200
    model_args.max_wav_length = 255995  # ~11.6s
    model_args.mel_norm_file = str(mel_path)
    model_args.dvae_checkpoint = str(dvae_path)
    config.model_args = model_args
    config.audio.d_vector_file = []

    # --- data ---
    config.datasets = [BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ft_voice",
        path=str(dataset),
        meta_file_train="metadata.csv",
        language=language,
    )]

    # --- hardware profile: keeps the desktop usable while training ---
    import hw as hwmod

    hw_info = hwmod.detect()
    if profile == "high" and hw_info.get("vram_total_gb", 0) < 10:
        print(f"[train] profile high needs 10GB+ VRAM "
              f"(have {hw_info.get('vram_total_gb')}) -> using mid")
        profile = "mid"
    prof_name = profile if profile != "auto" else hwmod.recommend(hw_info)
    prof = hwmod.apply(prof_name)
    print(f"[train] hardware profile: {prof_name} {prof}")
    try:
        free, _ = torch.cuda.mem_get_info(0)
        if free / 1e9 < 3.5:
            print(f"[train] WARNING: only {free/1e9:.1f} GB VRAM free, "
                  f"training needs ~3.8. Close browser/games first!")
    except Exception:
        pass

    # --- 4GB-friendly schedule ---
    config.output_path = str(out)
    config.run_name = run_name or config.run_name  # unique per job; same-minute retries collide otherwise
    config.batch_size = 1 if smoke else prof["batch_size"]
    config.batch_group_size = 48  # bucketing; 0 breaks the loader
    config.num_loader_workers = prof["num_workers"]
    config.num_eval_loader_workers = 0
    config.epochs = 5 if smoke else epochs
    config.optimizer = "AdamW"  # base config.json is inference-only (RAdam/None)
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8,
                               "weight_decay": 0.01}
    config.lr = 5e-6
    config.lr_scheduler = "NoamLR"
    config.lr_scheduler_params = {"warmup_steps": 1 if smoke else 50}
    config.save_step = 10 if smoke else save_step
    config.print_step = 1 if smoke else 5
    config.run_eval = False
    config.test_delay_epochs = 999 if smoke else 1
    # fp16 AMP overflows to NaN on this stack (verified); fp32 default.
    # Only --fp16 opts in; profiles never enable it implicitly.
    config.mixed_precision = bool(fp16)
    config.target_loss = "loss"
    config.print_eval = False
    first_wav = sorted((Path(dataset) / "wavs").glob("*.wav"))[0]
    config.test_sentences = [{
        "text": test_text,
        "speaker_wav": str(first_wav),
        "language": language,
    }]

    # --- samples (loaded once, reused every burst) ---
    train_samples, eval_samples = load_tts_samples(
        config.datasets, eval_split=eval_split, eval_split_size=0.05,
    )
    print(f"[train] {len(train_samples)} train / "
          f"{len(eval_samples) if eval_samples else 0} eval samples")

    def run_burst(resume_path, burst_epochs, tag):
        torch.backends.cudnn.benchmark = False  # determinism > speed here
        model = GPTTrainer.init_from_config(config)
        trainer = Trainer(
            TrainerArgs(
                restore_path=resume_path,
                skip_train_epoch=False,
                start_with_eval=False,
                grad_accum_steps=(1 if smoke else
                                  (grad_accum or prof["grad_accum"])),
                overfit_batch=smoke,
                small_run=5 if smoke else None,  # False truncates to 0 samples!
            ),
            config, str(out), model=model,
            train_samples=train_samples, eval_samples=eval_samples,
        )
        print(f"[train] burst {tag}: {burst_epochs} epoch(s)"
              + (f" from {resume_path}" if resume_path else " from base"))
        trainer.fit()

    def latest_checkpoint():
        cks = list(Path(out).rglob("checkpoint_*.pth"))
        def step(p):
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1
        return max(cks, key=step) if cks else None

    bursts = []
    if smoke or not burst_epochs:
        bursts = [(epochs if not smoke else 5, resume)]
    else:
        remaining, r = epochs, resume
        while remaining > 0:
            take = min(burst_epochs, remaining)
            bursts.append((take, r))
            remaining -= take
            r = "LATEST"  # resolved after each burst
    for i, (be, r) in enumerate(bursts):
        tag = f"{run_name or 'run'}-b{i}" if len(bursts) > 1 else (run_name or config.run_name)
        config.epochs = be
        config.run_name = tag
        config.save_step = 10 if smoke else save_step
        resume_path = r if r != "LATEST" else (
            str(latest_checkpoint()) if latest_checkpoint() else None)
        run_burst(resume_path, be, tag)
        if i < len(bursts) - 1:
            print(f"[train] burst {tag} done. Resting {rest_seconds}s "
                  f"so the PC stays usable (GPU idle)...")
            time.sleep(rest_seconds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="output of prepare_ft.py")
    p.add_argument("--language", default="ja")
    p.add_argument("--out", default="training/run")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-split", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="overfit 1 batch: VRAM feasibility check")
    p.add_argument("--resume", default=None)
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--test-text", default="こんにちは、これはテストです。",
                   help="sample synthesized after training")
    p.add_argument("--mel-stats", default=None,
                   help="mel_stats.pth; extracted from base model if missing")
    p.add_argument("--dvae-checkpoint", default=None,
                   help="dvae.pth; auto-downloaded from HuggingFace if missing")
    p.add_argument("--fp16", action="store_true",
                   help="enable fp16 AMP (NaNs on this stack; fp32 default)")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="override profile default")
    p.add_argument("--save-step", type=int, default=100)
    p.add_argument("--profile", default="auto", choices=["auto", "low", "mid", "high"],
                   help="hardware profile (see scripts/hw.py); auto detects")
    p.add_argument("--run-name", default=None,
                   help="unique run dir tag (web passes job id)")
    p.add_argument("--burst-epochs", type=int, default=0,
                   help="installments: train N epochs, rest, resume (0=off)")
    p.add_argument("--rest-seconds", type=int, default=90,
                   help="GPU idle gap between bursts so the PC stays usable")
    a = p.parse_args()
    train(Path(a.dataset), a.language, Path(a.out), a.epochs,
          a.eval_split, a.smoke, a.resume, a.base_dir, a.test_text,
          a.mel_stats, a.dvae_checkpoint, a.fp16, a.grad_accum,
          a.save_step, a.profile, a.run_name, a.burst_epochs,
          a.rest_seconds)
