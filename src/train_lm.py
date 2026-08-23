"""
Train a causal language model on a text dataset.

The script prepares the training data, optionally reorders model
embeddings, trains the language model, and evaluates the result.
"""

import math
import os

import numpy as np
import torch
import yaml

from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    enable_full_determinism,
    set_seed,
)

from utils.data_utils import (
    CustomDataCollatorForLanguageModeling,
    preprocess_lm_data,
)
from utils.lm_utils import (
    load_lm_model,
    load_lm_tokenizer,
    reorder_embeddings,
)
from utils.metrics import (
    compute_metrics,
    preprocess_logits_for_metrics,
)



# ============================================================
# Load configuration
# ============================================================

with open("configs/lm_train_config.yaml", "r",) as f:
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
np.random.seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load and prepare dataset
# ============================================================

DATASET_PATH = cfg["data"]["dataset_path"]

TRAIN_FILES = [
    os.path.join(
        DATASET_PATH,
        file,
    )
    for file in cfg["data"]["train_files"]
]


dataset = load_dataset(
    "text",
    data_files={"train": TRAIN_FILES},
)

# Split the training data into training and validation sets
dataset = dataset["train"].train_test_split(
    test_size=cfg["data"]["test_size"],
    shuffle=True,
    seed=SEED,
)


# ============================================================
# Load tokenizer
# ============================================================

LM_NAME = cfg["model"]["name"]

tokenizer_cfg = cfg["tokenizer"]


tokenizer = load_lm_tokenizer(
    tokenizer_type=tokenizer_cfg["type"],
    lm_name=LM_NAME,
    whisper_model=tokenizer_cfg["whisper_model"],
    whisper_path=tokenizer_cfg["whisper_path"],
    language=tokenizer_cfg.get("language"),
    task=tokenizer_cfg.get("task"),
)


# ============================================================
# Load language model
# ============================================================

model = load_lm_model(
    model_name=LM_NAME,
    model_path=cfg["model"]["path"],
    tokenizer=tokenizer,
    device=DEVICE,
)


# ============================================================
# Optionally reorder embeddings
# ============================================================

REORDER_TYPE = cfg["embedding"]["reorder_type"]

if REORDER_TYPE != "none":

    model = reorder_embeddings(
        model=model,
        whisper_tokenizer=tokenizer,
        model_name=LM_NAME,
        reorder_type=REORDER_TYPE,
    )


MODEL_OUTPUT_NAME = (
    f"{LM_NAME}"
    f"_tokenizer_{tokenizer_cfg['type']}"
    f"_reorder_{REORDER_TYPE}"
)

OUTPUT_PATH = os.path.join(
        OUTPUT_DIR, MODEL_OUTPUT_NAME,
)


# ============================================================
# Preprocess data
# ============================================================

dataset = dataset.map(
    lambda examples:
        preprocess_lm_data(
            examples,
            tokenizer,
            cfg["data"]["text_column"],
        ),
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
)


# ============================================================
# Load data collator
# ============================================================

data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)


# ============================================================
# Prepare training arguments
# ============================================================

train_cfg = cfg["training"]


training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,

    per_device_train_batch_size=
        train_cfg["train_batch_size"],

    per_device_eval_batch_size=
        train_cfg["eval_batch_size"],

    learning_rate=
        float(train_cfg["learning_rate"]),

    warmup_steps=
        train_cfg["warmup_steps"],

    num_train_epochs=
        train_cfg["epochs"],

    gradient_checkpointing=
        train_cfg["gradient_checkpointing"],

    fp16=train_cfg["fp16"],

    weight_decay=
        train_cfg["weight_decay"],

    save_strategy=
        train_cfg["save_strategy"],

    logging_strategy=
        train_cfg["logging_strategy"],

    save_total_limit=
        train_cfg["save_total_limit"],

    evaluation_strategy="epoch",

    load_best_model_at_end=True,

    greater_is_better=False,

    seed=SEED,

    report_to=["tensorboard"],

    push_to_hub=False,
)


# ============================================================
# Create trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],

    data_collator=data_collator,

    compute_metrics=lambda pred:
        compute_metrics(
            pred,
            tokenizer,
        ),

    preprocess_logits_for_metrics=
        preprocess_logits_for_metrics,

    tokenizer=tokenizer,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=
                train_cfg["early_stopping_patience"]
        )
    ],
)


# ============================================================
# Train model
# ============================================================

trainer.train()


# ============================================================
# Save and evaluate model
# ============================================================

if cfg["output"]["save_model"]:
    trainer.save_model(OUTPUT_PATH)

results = trainer.evaluate()

print(results)

# Compute perplexity from the evaluation loss
print(f"Eval loss: {results['eval_loss']}")
print(f"Perplexity: {math.exp(results['eval_loss'])}")