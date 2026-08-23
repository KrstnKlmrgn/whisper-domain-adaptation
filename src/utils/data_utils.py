"""
Utilities for data preparation, collation, and audio processing.

This module provides functions for preparing audio, text, and
spectrogram data for Whisper and causal language model training,
as well as custom data collators and audio/spectrogram helpers.
"""


import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import (
    _torch_collate_batch,
    pad_without_fast_tokenizer_warning,
)


# ============================================================
# Preprocessing utils
# ============================================================


def prepare_dataset(
    example,
    processor,
    audio_column,
    text_column,
    sampling_rate,
    skip_audio=False,
    num_mel_bins=None,
):
    """
    Convert audio into Whisper input features
    and text into token labels.

    If skip_audio=True, no log-Mel features are computed.
    Instead, zero-valued placeholder features are created.
    These placeholders are ignored later when the encoder outputs
    are replaced by dummy representations in the custom trainer.
    """

    # Prepare encoder input

    if skip_audio:

        if num_mel_bins is None:
            raise ValueError(
                "num_mel_bins must be provided when skip_audio=True."
            )
        # Placeholder features. These are never used by the model,
        # since the encoder outputs are replaced in the custom trainer
        example["input_features"] = np.zeros(
            (num_mel_bins, 3000), dtype=np.float32,
        )

    else:

        # Load and resample audio data to 16kHz
        audio = example[audio_column]

        # Compute log-Mel input features from input audio array
        example["input_features"] = (
            processor.feature_extractor(
                audio["array"],
                sampling_rate=sampling_rate,
            )
            .input_features[0]
        )

    # Prepare decoder targets

    # Encode target text to label ids
    example["labels"] = (
        processor.tokenizer(
            example[text_column].replace("_", " ")
        )
        .input_ids
    )

    return example



def preprocess_lm_data(
    examples,
    tokenizer,
    text_column,
):
    """
    Tokenize text for causal LM training/evaluation.
    """

    # Remove empty lines
    examples[text_column] = [
        text
        for text in examples[text_column]
        if len(text) > 0 and not text.isspace()
    ]

    tokens = tokenizer(
        examples[text_column],
        padding=False,
        add_special_tokens=True,
        truncation=False,
        return_special_tokens_mask=True,
    )

    examples["input_ids"] = tokens.input_ids
    examples["attention_mask"] = tokens.attention_mask
    examples["special_tokens_mask"] = tokens.special_tokens_mask

    return examples



def prepare_spectrogram_dataset(
    example,
    tts_model,
    enhancer,
    processor,
    text_column,
    normalization="minmax",
    enhancer_seed=42,
):
    """
    Generate Whisper input features from text using a TTS model.

    Pipeline:
        text
        -> TTS spectrogram
        -> optional spectrogram enhancement
        -> optional normalization
        -> padding to Whisper length
        -> Whisper input features
        -> tokenized labels
    """

    text = example[text_column].replace("_", " ")

    # Generate spectrogram
    with torch.no_grad():
        spectrogram = tts_model.generate(text)

    # Optional spectrogram enhancement
    if enhancer is not None:
        with torch.no_grad():
            spectrogram = tts_model.enhance(
                enhancer, spectrogram, seed=enhancer_seed,
            )

    # Optional normalization
    if normalization == "minmax":
        spectrogram = minmax_normalize_spectrogram(spectrogram)

    elif normalization == "whisper":
        spectrogram = whisper_like_normalize_spectrogram(spectrogram)

    elif normalization == "none":
        pass

    else:
        raise ValueError(
            f"Unknown spectrogram normalization: {normalization}"
        )

    # Pad to Whisper encoder length
    spectrogram = pad_spectrogram_to_length(spectrogram)

    # Whisper encoder input
    example["input_features"] = (
        spectrogram.squeeze(0).cpu().numpy()
    )

    # Prepare decoder targets
    example["labels"] = (
        processor.tokenizer(text).input_ids
    )

    return example



# ============================================================
# Data Collators
# ============================================================

class DataCollatorSpeechSeq2SeqWithPadding:

    def __init__(
        self,
        processor,
        decoder_start_token_id,
    ):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id


    def __call__(
        self,
        features: List[
            Dict[str, Union[List[int], torch.Tensor]]
        ],
    ):
        # Split inputs and labels because they have to be of 
        # different lengths and require different padding methods.

        # Return torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        # Pad input features using the Whisper feature extractor
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt",
        )

        # Get tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        # Pad labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt",
        )

        # Replace padding with -100 so they are ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100,
        )

        # Remove the decoder start token because the model adds it internally
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels


        return batch


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def torch_call(
        self,
        examples: List[
            Union[List[int],Any,Dict[str, Any]]
            ]
        ) -> Dict[str, Any]:

        # Handle dict or lists with proper padding and conversion to tensor
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of
            )

        else:
            batch = {
                "input_ids": _torch_collate_batch(
                        examples,
                        self.tokenizer,
                        pad_to_multiple_of=self.pad_to_multiple_of
                        )
            }

        # Remove the special token mask after it has been used for padding
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                    batch["input_ids"],
                    special_tokens_mask=special_tokens_mask,
            )

        else:
            # Create labels from the input IDs
            labels = batch["input_ids"].clone()

            if self.tokenizer.pad_token_id is not None:
                # Ignore padding tokens when computing the loss
                # Do not mask the first eos token!
                labels = labels.masked_fill(
                    batch.attention_mask.ne(1), -100
                    )

            batch["labels"] = labels


        return batch


# ============================================================
# Spectrogram Helpers
# ============================================================

def save_spectrogram_as_image(spectrogram, output_dir, wav_name):
     """
     Save a spectrogram as a grayscale image.
     """

    with torch.no_grad():
        spectrogram_np = (
            spectrogram.squeeze(0).detach().cpu().numpy()
        )

    plt.figure(figsize=(10, 4))

    plt.imshow(
        spectrogram_np,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    plt.colorbar(label="Amplitude")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()

    spec_name = wav_name.rsplit(".", 1)[0] + ".png"
    output_path = os.path.join(output_dir, spec_name)

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()


def minmax_normalize_spectrogram(spectrogram):
    """
    Normalize spectrogram values to the range [-1, 1]
    using min-max scaling.

    Expected input shape:
        [batch, mel_bins, time]
        or
        [mel_bins, time]
    """

    min_val = torch.min(spectrogram)
    max_val = torch.max(spectrogram)

    if max_val == min_val:
        return torch.zeros_like(spectrogram)

    # min-max scaling to range [-1, 1]
    normalized_spectrogram = (
        2 * (spectrogram - min_val) / (max_val - min_val)- 1
    )

    return normalized_spectrogram


def whisper_like_normalize_spectrogram(spectrogram):
    """
    Apply Whisper-style normalization to a log-Mel spectrogram.

    This follows the normalization used in the Whisper feature extractor:
    1. Limit dynamic range by clipping values below max - 8.
    2. Shift and scale values.

    Expected input:
        log-Mel spectrogram
        [batch, mel_bins, time]
        or
        [mel_bins, time]
    """

    max_val = torch.max(spectrogram)

    spectrogram = torch.clamp(
        spectrogram, max=max_val - 8.0,
    )
    spectrogram = (spectrogram + 4.0) / 4.0

    return spectrogram


def pad_spectrogram_to_length(spectrogram,target_length=3000):
    """
    Pad spectrogram in time dimension to target length.

    Expected input shape:
        [batch, mel_bins, time]
        or
        [mel_bins, time]
    """

    if spectrogram.dim() == 3:
        spectrogram = spectrogram.squeeze(0)

    current_length = spectrogram.shape[-1]

    # Check if padding is needed
    if current_length < target_length:

        # Calculate the required padding for the last dimension
        padding_amount = target_length - current_length

        # Pad the tensor to the right (only in the last dimension)
        spectrogram = F.pad(
            spectrogram,
            (0, padding_amount),
            mode="constant",
            value=0.0,
        )

    elif current_length > target_length:

        # spectrogram = spectrogram[..., :target_length]
        print(
            f"Warning: spectrogram length {current_length} > {target_length}"
        )

    return spectrogram



# ============================================================
# Audio Helpers
# ============================================================


def resample_audio(
    audio,
    original_sampling_rate,
    target_sampling_rate
):

    if original_sampling_rate == target_sampling_rate:
        return audio

    return librosa.resample(
        audio,
        orig_sr=original_sampling_rate,
        target_sr=target_sampling_rate
    )


def save_audio(audio, output_dir, wav_name, samplerate):

    sf.write(
        output_dir + wav_name,
        audio,
        samplerate=samplerate,
        format="WAV"
    )