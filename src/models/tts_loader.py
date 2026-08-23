"""
Loads TTS models and spectrogram enhancers.

The loader selects the TTS model specified in the configuration and
optionally loads a pretrained spectrogram enhancement model.

"""

from models.fastpitch import FastPitchTTS
from models.speecht5 import SpeechT5TTS

from nemo.collections.tts.models import SpectrogramEnhancerModel



def load_tts_model(config, device):

    model_type = config["tts"]["model"]

    if model_type == "fastpitch":
        tts_model = FastPitchTTS(
            config,
            device
        )
    elif model_type == "speecht5":
        tts_model = SpeechT5TTS(
            config,
            device
        )
    else:
        raise ValueError(f"Unknown TTS model: {model_type}")


    enhancer = None

    if config["tts"]["use_enhancer"]:
        enhancer = SpectrogramEnhancerModel.from_pretrained(
            model_name=config["enhancer"]["model_name"]
        )
        enhancer.to(device)
        enhancer.eval()


    return tts_model, enhancer


