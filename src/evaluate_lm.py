"""
Evaluate a causal language model on a text dataset.

The script tokenizes the evaluation data, computes the negative
log-likelihood and perplexity, and optionally saves the results.
"""

import math
import os

import pandas as pd
import torch
import yaml

from datasets import Dataset
from transformers import (
    enable_full_determinism,
    set_seed,
)

from utils.lm_utils import (
    load_lm_model,
    load_lm_tokenizer,
    preprocess_lm_data,
)
from utils.metrics import compute_ppl



# ============================================================
# Load configuration
# ============================================================

with open("configs/lm_evaluation_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR = config["output"]["path"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


SEED = cfg["seed"]
set_seed(SEED)
enable_full_determinism(SEED)
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Data
DATASET_PATH = cfg["data"]["dataset_path"]
TEST_SET = cfg["data"]["test_set"]


# LM
LM = cfg["model"]["name"]
LM_PATH = cfg["model"]["path"]



# ============================================================
# Load tokenizer
# ============================================================

# Tokenizer
TOKENIZER_TYPE = cfg["tokenizer"]["type"]


tokenizer = load_lm_tokenizer(
    tokenizer_type=TOKENIZER_TYPE,
    lm_name=LM,
    whisper_model=cfg["tokenizer"]["whisper_model"],
    whisper_path=cfg["tokenizer"]["whisper_path"],
    language=cfg["tokenizer"]["language"],
    task=cfg["tokenizer"]["task"],
)


# ============================================================
# Load LM
# ============================================================

lm = load_lm_model(
    model_name=LM,
    model_path=LM_PATH,
    tokenizer=tokenizer,
    device=DEVICE,
)

lm.eval()


# ============================================================
# Load dataset
# ============================================================

test_df = pd.read_csv(
    DATASET_PATH + TEST_SET
)

dataset = Dataset.from_pandas(
    test_df
)


# ============================================================
# Helper functions
# ============================================================

# Store batch losses for corpus-level perplexity
losses = []


def preprocess_data(examples):

    return preprocess_lm_data(
        examples,
        tokenizer,
        cfg["data"]["text_column"],
    )


def inference(examples):

    batch = tokenizer.pad(
        {
            "input_ids": examples["input_ids"],
            "attention_mask": examples["attention_mask"],
            "special_tokens_mask": examples["special_tokens_mask"],
        },
        return_tensors="pt",
    )

    input_ids = batch["input_ids"].to(DEVICE)

    attention_mask = (
        batch["attention_mask"].to(DEVICE)
    )

    labels = input_ids.clone()

    # Ignore padding
    labels[attention_mask.ne(1)] = -100

    # Optionally ignore whisper special tokens
    #labels[special_tokens_mask == 1] = -100

    with torch.no_grad():
        outputs = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = outputs.loss
    losses.append(loss.detach())

    examples["NegLogLikelihood"] = (
        [loss.item()] * len(examples["input_ids"])
    )
    examples["PPL"] = (
        [math.exp(loss.item())] * len(examples["input_ids"])
    )

    return examples


# ============================================================
# Run inference
# ============================================================

REMOVE_COLUMNS = [
    column
    for column in ["audio", "duration"]
    if column in dataset.column_names
]


dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=REMOVE_COLUMNS,
)


dataset = dataset.map(
    inference,
    batch_size=cfg["evaluation"]["batch_size"],
    batched=True,
)


# ============================================================
# Evaluate
# ============================================================

ppl = compute_ppl(losses)

print(f"Perplexity: {ppl.item()}")


# ============================================================
# Save results
# ============================================================

if cfg["output"]["save_predictions"]:

    FILENAME = (
        f"eval_"
        f"{LM}_"
        f"{TOKENIZER_TYPE}.csv"
    )

    dataset.to_csv(
        os.path.join(OUTPUT_DIR, FILENAME),
        columns=[
        "sentence",
        "NegLogLikelihood",
        "PPL",
        ]
    )