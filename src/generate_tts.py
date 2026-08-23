"""
Generate synthetic speech data using a TTS model.

The script loads a TTS model and an optional spectrogram enhancer,
generates spectrograms from text, optionally enhances and saves them,
and optionally converts the spectrograms to audio using a vocoder.
"""

import os

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (
    enable_full_determinism,
    set_seed,
)

from tts_loader import load_tts_model
from utils.data_utils import (
    save_audio,
    save_spectrogram_as_image,
)


# ============================================================
# Load configuration
# ============================================================

with open("configs/tts_config.yaml") as f:
    config = yaml.safe_load(f)


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR = config["output"]["path"]

os.makedirs(OUTPUT_DIR,exist_ok=True)


SEED = config["seed"]

set_seed(SEED)
enable_full_determinism(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load TTS model and enhancer
# ============================================================

tts_model, enhancer = load_tts_model(
    config, DEVICE
)


# ============================================================
# Helper functions
# ============================================================

def get_wav_name(sample, idx):

    audio = sample[config["data"]["audio_column"]]

    if isinstance(audio, str):
        return audio.split("/")[-1]

    if isinstance(audio, dict):

        if audio.get("path") is not None:
            return audio["path"].split("/")[-1]

        return f"sample_{idx}.wav"

    return f"sample_{idx}.wav"


def add_suffix(filename, suffix):

    name, ext = filename.rsplit(".", 1)

    return f"{name}{suffix}.{ext}"


# ============================================================
# Generate audio
# ============================================================

def generate(sample, idx):

    sentence = sample[config["data"]["text_column"]]
    wav_name = get_wav_name(sample, idx)

    with torch.no_grad():

        # Generate a mel-spectrogram from the input text
        spectrogram = tts_model.generate(sentence)


        # Optionally enhance the generated spectrogram
        if enhancer is not None:
            spectrogram = tts_model.enhance(
                enhancer, spectrogram
            )
            wav_name = add_suffix(wav_name, "_enh")


        # Save spectrogram
        if config["output"]["save_spectrogram"]:
            save_spectrogram_as_image(
                spectrogram,
                OUTPUT_DIR,
                wav_name
            )


        # Convert the spectrogram to audio if a vocoder is enabled
        if config["tts"]["use_vocoder"]:
            audio = tts_model.vocode(spectrogram)

            if config["output"]["save_audio"]:
                '''
                audio = resample_audio(
                    audio,
                    tts_model.sample_rate,
                    config["data"]["sampling_rate"]
                )'''

                save_audio(
                    audio,
                    OUTPUT_DIR,
                    wav_name,
                    tts_model.sample_rate
                    #config["data"]["sampling_rate"]
                )

    return sample


# ============================================================
# Load dataset
# ============================================================

DATASET_PATH = config["data"]["dataset_path"]
DATASET_FILE = config["data"]["dataset_file"]


df = pd.read_csv(DATASET_PATH + DATASET_FILE)

dataset = Dataset.from_pandas(df)


# ============================================================
# Run generation
# ============================================================

dataset.map(
    generate,
    with_indices=True
)
