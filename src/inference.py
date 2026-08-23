"""
Perform ASR inference using Whisper models with optional language model fusion.

This script loads a Whisper model, optionally adds shallow or deep fusion
with a language model, performs transcription on a test dataset, and computes
WER and perplexity.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import (
    Audio,
    Dataset,
    load_dataset,
)
from safetensors.torch import load_file
from transformers import (
    GenerationConfig,
    GPT2LMHeadModel,
    WhisperConfig,
    WhisperForConditionalGeneration,
    enable_full_determinism,
    set_seed,
)

from models.custom_whisper_generation_mixin import (
    CustomWhisperGenerationMixin
)
from models.deep_fusion import DeepFusionWhisper
from models.lm_scorer import LMScorer
from utils.metrics import (
    compute_ppl,
    compute_wer,
)
from utils.whisper_utils import (
    configure_whisper_generation,
    load_whisper_processor,
)


# ============================================================
# Load configuration
# ============================================================

with open("configs/inference_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR = cfg["output"]["path"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


SEED = cfg["seed"]

set_seed(SEED)
enable_full_determinism(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Data

DATASET_PATH = cfg["data"]["dataset_path"]

TEST_SET = cfg["data"]["test_set"]

AUDIO_COLUMN = cfg["data"]["audio_column"]
TEXT_COLUMN = cfg["data"]["text_column"]


# Whisper

LANGUAGE = cfg["whisper"]["language"]
TASK = cfg["whisper"]["task"]

WHISPER_MODEL = cfg["whisper"]["model"]
WHISPER_PATH = cfg["whisper"]["path"]

MODEL = WHISPER_MODEL

MODEL_OUTPUT_NAME = f"transcr-{MODEL}.csv"


# Generation

NUM_BEAMS = cfg["generation"]["num_beams"]


# Model type

MODEL_TYPE = cfg["model"]["type"]


# ============================================================
# Load Whisper model and processor
# ============================================================

model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_PATH + MODEL,
    #tie_word_embeddings=False,
    #ignore_mismatched_sizes=False
)

processor, forced_decoder_ids = load_whisper_processor(
    whisper_path=WHISPER_PATH,
    whisper_model=WHISPER_MODEL,
    language=LANGUAGE,
    task=TASK,
)


# ============================================================
# Add Shallow Fusion model
# ============================================================

if MODEL_TYPE == "shallow_fusion":

    if NUM_BEAMS <= 1:
        raise ValueError(
            "Shallow fusion requires beam search, but num_beams=1 was set. "
            "Please set num_beams > 1."
        )

    LM = cfg["shallow_fusion"]["lm"]["name"]
    LM_PATH = cfg["shallow_fusion"]["lm"]["path"]
    LM_COEFFICIENT = cfg["shallow_fusion"]["weight"]

    MODEL_OUTPUT_NAME = (
        f"transcr-{MODEL}_shallow_fusion"
        f"_beams{NUM_BEAMS}"
        f"_lmco{LM_COEFFICIENT}_{LM}.csv"
    )


    lm = GPT2LMHeadModel.from_pretrained(LM_PATH + LM)

    if lm.transformer.wte.num_embeddings != len(processor.tokenizer):
        lm.resize_token_embeddings(len(processor.tokenizer))

    lm.eval()
    lm.to(DEVICE)

    scorer = LMScorer(
        lm, processor.tokenizer,
    )

    model.lm_scorer = scorer
    model.lm_weight = LM_COEFFICIENT

    model.__class__ = type(
        "CustomWhisper",
        (
            CustomWhisperGenerationMixin,
            model.__class__,
        ),
        {}
    )


# ============================================================
# Add Deep Fusion model
# ============================================================

if MODEL_TYPE == "deep_fusion":

    LM = cfg["deep_fusion"]["lm"]["name"]
    LM_PATH = cfg["deep_fusion"]["lm"]["path"]

    DEEP_FUSION_MODEL = (cfg["deep_fusion"]["model"]["name"])
    DEEP_FUSION_PATH = (cfg["deep_fusion"]["model"]["path"])

    MODEL_OUTPUT_NAME = (
        f"transcr-{MODEL}_{DEEP_FUSION_MODEL}.csv"
    )


    lm = GPT2LMHeadModel.from_pretrained(LM_PATH + LM)

    lm.eval()
    lm.to(DEVICE)

    scorer = LMScorer(
        lm,
        processor.tokenizer,
    )

    whisper_config = WhisperConfig.from_pretrained(
        DEEP_FUSION_PATH + DEEP_FUSION_MODEL
    )

    generation_config = GenerationConfig.from_pretrained(
        DEEP_FUSION_PATH + DEEP_FUSION_MODEL
    )

    # Disable weight tying.
    # Otherwise, from_pretrained() may tie the output layer to the Whisper embeddings,
    # which have a different input dimension than the Deep Fusion output layer.
    model.config.tie_word_embeddings = False

    model = DeepFusionWhisper(
        pretrained_whisper=model,
        lm_scorer=scorer,
        use_lm=cfg["deep_fusion"]["use_lm"],
        copy_weights=cfg["deep_fusion"]["copy_weights"],
    )

    weights_path = (
        DEEP_FUSION_PATH
        + DEEP_FUSION_MODEL
        + "/model.safetensors"
    )

    state_dict = load_file(weights_path)

    model.load_state_dict(state_dict)

    model.generation_config = generation_config
    model.config = whisper_config


# ============================================================
# Configure Whisper
# ============================================================

configure_whisper_generation(
    model,
    WHISPER_MODEL,
    LANGUAGE,
    TASK,
    forced_decoder_ids=forced_decoder_ids,
    max_length=cfg["generation"]["max_length"],
    num_beams=NUM_BEAMS,
    return_timestamps=False,
)

model.to(DEVICE)


# ============================================================
# Load dataset
# ============================================================

test_df = pd.read_csv(DATASET_PATH + TEST_SET)

dataset = Dataset.from_pandas(test_df)

'''
# ATCO2-Test

dataset = load_dataset(
    "Jzuluaga/atco2_corpus_1h",
    split='test[:1]'
)

dataset = dataset.remove_columns([
    'id',
    'segment_start_time',
    'segment_end_time'
])

dataset = dataset.rename_column(
    "text",
    TEXT_COLUMN
)
#'''
'''
# FLEURS (100 samples)

dataset = load_dataset(
    "google/fleurs",
    "en_us",
    split="test[:1]"
)

dataset = dataset.remove_columns([
    'id',
    'num_samples',
    'path',
    'raw_transcription',
    'gender',
    'lang_id',
    'language',
    'lang_group_id'
])

dataset = dataset.rename_column(
    "transcription",
    TEXT_COLUMN
)
#'''


# ============================================================
# Prepare dataset
# ============================================================

SAMPLING_RATE = cfg["data"]["sampling_rate"]

# Convert audio
dataset = dataset.cast_column(
    AUDIO_COLUMN,
    Audio(sampling_rate=SAMPLING_RATE)
)

def prepare_whisper_labels(input_ids):
    # Pad the labels to max length
    labels_batch = processor.tokenizer.pad(
        input_ids,
        return_tensors="pt",
    )

    # Replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100,
    )

    # If bos token is appended in previous tokenization step,
    # cut bos token here as it's appended later anyways
    if (labels[:, 0] == model.config.decoder_start_token_id).all().cpu().item():
        labels = labels[:, 1:]

    return labels


# ============================================================
# Run inference
# ============================================================

losses = []
times = []


def inference(batch):

    # Prepare input features

    arrays = [
        x["array"]
        for x in batch[AUDIO_COLUMN]
    ]

    input_features = (
        processor.feature_extractor(
            arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        .input_features
        .to(DEVICE)
    )

    # Generate transcriptions

    with torch.no_grad():

        start = time.time()

        predicted_ids = model.generate(
            input_features,
            generation_config=model.generation_config,
            num_beams=NUM_BEAMS,
        )

        end = time.time()
        times.append(end - start)

    # Decode predictions for WER computation
    results = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        normalize=True,
    )

    # Decode predictions without normalization
    unnorm_results = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        normalize=False,
    )

    # Prepare reference labels

    labels = processor.tokenizer(
        [
            sent.replace("_", " ")
            for sent in batch[TEXT_COLUMN]
        ],
        padding=False,
        add_special_tokens=True,
        truncation=False,
        return_special_tokens_mask=True
    )

    labels = prepare_whisper_labels(labels)

    # Compute loss

    with torch.no_grad():

        outputs = model(
            input_features,
            labels=labels.to(input_features.DEVICE),
        )

    loss = (
        outputs["loss"]
        if isinstance(outputs, dict)
        else outputs[0]
    )

    losses.append(loss.mean().detach())

    # Store predictions

    batch["predictions"] = results

    batch["references"] = [
        processor.tokenizer._normalize(text)
        for text in batch[TEXT_COLUMN]
    ]

    batch["unnorm_predictions"] = unnorm_results

    return batch


dataset = dataset.map(
    inference,
    batch_size=cfg["inference"]["batch_size"],
    batched=True,
)


# ============================================================
# Evaluate predictions
# ============================================================

# Compute test perplexity
ppl = compute_ppl(losses)

# Compute WER
wer_score = compute_wer(
    dataset["predictions"],
    dataset["references"],
)

# ============================================================
# Save results
# ============================================================

dataset = dataset.add_item(
    {"references": f"WER: {wer_score:.2f} %"}
)

dataset = dataset.add_item(
    {"references": f"PPL: {ppl}"}
)

dataset.to_csv(
    os.path.join(
        OUTPUT_DIR, MODEL_OUTPUT_NAME,
    ),
    columns=[
        "references",
        "predictions",
        "unnorm_predictions",
    ],
)

print(f"mean inference time: {np.mean(times)}")