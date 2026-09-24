import soundfile as sf
import torch
import warnings
warnings.filterwarnings("ignore")

try:
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT, UNetT
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # standard signature for load_model usually includes model_name or similar. Let's see what works
    model_obj = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path="hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
        mel_spec_type="vocos",
        vocab_file="",
    )
    vocoder = load_vocoder()
    
    audio, sample_rate, spec = infer_process(
        ref_audio="data/audio_clean/deer_expressive.wav",
        ref_text="",
        gen_text="Hello world, this is a test of F5 TTS.",
        model_obj=model_obj,
        vocoder=vocoder,
        device=device
    )
    
    sf.write("outputs/f5_test.wav", audio, sample_rate)
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
