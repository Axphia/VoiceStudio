"""Persistent voice-cloning server: model loads ONCE, clones are instant.

Latents are cached per reference wav under data/audio_clean/*.pt,
so repeat clones skip both model load AND conditioning compute.

Usage:
  uv run python scripts/server.py --share
  # open http://127.0.0.1:7860
"""
import argparse
import tempfile
import time
import warnings
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS.*")

import gradio as gr
import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LATENT_DIR = Path("data/audio_clean")

_model = None
_latents_cache: dict[str, dict] = {}


def get_model():
    global _model
    if _model is None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.perf_counter()
        _model = TTS(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model load: {time.perf_counter() - t0:.1f}s")
    return _model


def get_latents(speaker_wav: str) -> dict:
    if speaker_wav not in _latents_cache:
        model = get_model()
        dev = next(model.synthesizer.tts_model.parameters()).device
        pt = LATENT_DIR / (Path(speaker_wav).stem + ".pt")
        if pt.exists():
            print(f"Latents cache hit: {pt}")
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
        else:
            print(f"Computing latents for {speaker_wav} ...")
            xtts = model.synthesizer.tts_model
            gpt, spk = xtts.get_conditioning_latents(audio_path=speaker_wav)
            d = {"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}
            torch.save(d, str(pt))
            print(f"Latents cached -> {pt}")
        _latents_cache[speaker_wav] = {
            "gpt_cond_latent": d["gpt_cond_latent"].to(dev),
            "speaker_embedding": d["speaker_embedding"].to(dev),
        }
    return _latents_cache[speaker_wav]


def list_refs():
    wavs = sorted(LATENT_DIR.glob("*.wav"))
    return [str(w) for w in wavs] or ["(no .wav in data/audio_clean — run extract first)"]


def clone(text, speaker_wav, language, temperature, speed, engine_name="xtts"):
    if not text or not text.strip():
        raise gr.Error("Type some text first.")
    if engine_name != "xtts":
        from engines.registry import get_engine

        out = get_engine(engine_name).synthesize(
            text.strip(), Path(tempfile.mktemp(suffix=".wav")), language,
            ref_audio=Path(speaker_wav), speed=speed,
        )
        return str(out)
    model = get_model()
    lat = get_latents(speaker_wav)
    with torch.inference_mode():
        wav = model.synthesizer.tts_model.inference(
            text.strip(), language,
            lat["gpt_cond_latent"], lat["speaker_embedding"],
            temperature=temperature, speed=speed,
            enable_text_splitting=True,
        )["wav"]
    tmp = tempfile.mktemp(suffix=".wav")
    model.synthesizer.save_wav(wav, tmp)
    return tmp


def build_ui():
    from engines.registry import list_engines

    refs = list_refs()
    engine_names = sorted(list_engines()) or ["xtts"]
    with gr.Blocks(title="VoiceStudio") as demo:
        gr.Markdown("# VoiceStudio — zero-shot cloning (model stays loaded)")
        with gr.Row():
            text = gr.Textbox(label="Text", lines=3, placeholder="えっ、本当に？信じられないよ！")
            ref = gr.Dropdown(choices=refs, value=refs[0], label="Reference voice")
        with gr.Row():
            engine = gr.Dropdown(choices=engine_names, value="xtts", label="Engine")
            lang = gr.Dropdown(choices=["ja", "en"], value="ja", label="Language")
            temp = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="Temperature (expressiveness)")
            speed = gr.Slider(0.8, 1.2, value=0.95, step=0.05, label="Speed")
        btn = gr.Button("Clone", variant="primary")
        audio = gr.Audio(label="Output")
        btn.click(clone, [text, ref, lang, temp, speed, engine], audio)
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    get_model()  # warm up before serving
    build_ui().launch(server_port=args.port, share=args.share)
