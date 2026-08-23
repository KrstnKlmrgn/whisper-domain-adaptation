"""
Fine-tune Whisper models for ASR using audio or text-based inputs.

This script loads supports standard Whisper fine-tuning as well as extensions for
Deep Fusion and ILMA. Depending on the configuration, the script
loads the corresponding model, prepares the dataset, trains the
model, and saves the resulting checkpoint.
"""

import os

import pandas as pd
import torch
import yaml
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    load_dataset,
)
from transformers import (
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    enable_full_determinism,
    set_seed,
)
from models.deep_fusion import DeepFusionWhisper
from models.lm_scorer import LMScorer
from models.tts_loader import load_tts_model
from utils.data_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    prepare_dataset,
    prepare_spectrogram_dataset,
)
from utils.metrics import compute_metrics
from utils.whisper_utils import (
    configure_whisper_generation,
    freeze_decoder,
    freeze_encoder,
    load_whisper_processor,
)


# ============================================================
# Load configuration
# ============================================================

with open("configs/whisper_train_config.yaml") as f:
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

MODEL_TYPE = cfg["model"]["type"]


# ============================================================
# Load Whisper model and processor
# ============================================================

WHISPER_MODEL = cfg["whisper"]["model"]
WHISPER_PATH = cfg["whisper"]["path"]

model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_PATH + WHISPER_MODEL
)

processor, forced_decoder_ids = load_whisper_processor(
    whisper_path=WHISPER_PATH,
    whisper_model=WHISPER_MODEL,
    language=cfg["whisper"]["language"],
    task=cfg["whisper"]["task"],
)

MODEL_OUTPUT_NAME = (
    f"whisper_finetuned_{WHISPER_MODEL}_{cfg["experiment"]["name"]}"
)

# ============================================================
# Add Deep Fusion model
# ============================================================

if MODEL_TYPE == "deep_fusion":

    # Load LM
    lm = GPT2LMHeadModel.from_pretrained(
        cfg["deep_fusion"]["lm"]["path"]
        +
        cfg["deep_fusion"]["lm"]["name"]
    )

    lm.eval()
    lm.to(DEVICE)

    scorer = LMScorer(
        lm, processor.tokenizer,
    )

    # Keep the original Whisper model for Deep Fusion
    whisper_model = model

    # Create Deep Fusion model
    model = DeepFusionWhisper(
        pretrained_whisper=whisper_model,
        lm_scorer=scorer,
        use_lm=cfg["deep_fusion"]["use_lm"],
        copy_weights=cfg["deep_fusion"]["copy_weights"],
        gate_type=cfg["deep_fusion"]["gate_type"],
    )

    MODEL_OUTPUT_NAME = (
        f"whisper_finetuned_deep_fusion"
        f"_{cfg['experiment']['name']}"
        f"_{cfg['deep_fusion']['lm']['name']}"
        f"_{cfg['deep_fusion']['use_lm']}"
        f"_{cfg['deep_fusion']['copy_weights']}"
    )


# ============================================================
# Configure ILMA
# ============================================================

if MODEL_TYPE == "ilma":

    # Set dummy encoder generator
    dummy_generator = torch.Generator(device=DEVICE)
    dummy_generator.manual_seed(cfg["ilma"]["generator_seed"])

    MODEL_OUTPUT_NAME = (
        f"whisper-{WHISPER_MODEL}"
        f"_ilma"
        f"_{cfg['ilma']['dummy_input']}"
        f"_std{cfg['ilma']['dummy_std']}"
        f"_seed{cfg['ilma']['generator_seed']}"
    )


# ============================================================
# Configure Whisper
# ============================================================

# Freeze encoder
if cfg["freeze"]["encoder"]:
    freeze_encoder(model)

# Freeze decoder
if cfg["freeze"]["decoder"]:
    freeze_decoder(model)

# Freeze cross-attention in decoder
if cfg["freeze"]["cross_attention"]:
    freeze_cross_attention(model)

# Freeze decoder except embeddings
if cfg["freeze"]["decoder_except_embeddings"]:
    freeze_decoder_except_embeddings(model)


configure_whisper_generation(
    model,
    WHISPER_MODEL,
    cfg["whisper"]["language"],
    cfg["whisper"]["task"],
    forced_decoder_ids=None,
    max_length=cfg["training"]["generation_max_length"],
)

model.config.use_cache = False
model.config.forced_decoder_ids = None

model.to(DEVICE)


# ============================================================
# Load dataset
# ============================================================

data_cfg = cfg["data"]

dataset = DatasetDict()

train_df = pd.read_csv(
    data_cfg["dataset_path"]
    + data_cfg["train_set"]
)
val_df = pd.read_csv(
    data_cfg["dataset_path"]
    + data_cfg["validation_set"]
)

dataset["train"] = Dataset.from_pandas(train_df)
dataset["validation"] = Dataset.from_pandas(val_df)

'''
dataset["train"] = load_dataset(
    "google/fleurs",
    "en_us",
    split="train[:5]"
)

dataset["validation"] = load_dataset(
    "google/fleurs",
    "en_us",
    split="validation[:5]"
)

dataset = dataset.remove_columns([
    "id",
    "num_samples",
    "path",
    "raw_transcription",
    "gender",
    "lang_id",
    "language",
    "lang_group_id"
])

dataset = dataset.rename_column(
    "transcription",
    data_cfg["text_column"]
)
'''


# ============================================================
# Prepare dataset
# ============================================================

INPUT_TYPE = data_cfg["input_type"]

if INPUT_TYPE == "audio":

    if MODEL_TYPE == "ilma":
        prepare_fn = lambda example: prepare_dataset(
            example,
            processor,
            data_cfg["audio_column"],
            data_cfg["text_column"],
            data_cfg["sampling_rate"],
            skip_audio=True,
            num_mel_bins=model.config.num_mel_bins,
        )

    else:
        dataset = dataset.cast_column(
            data_cfg["audio_column"],
            Audio(sampling_rate=data_cfg["sampling_rate"]),
        )

        prepare_fn = lambda example: prepare_dataset(
            example,
            processor,
            data_cfg["audio_column"],
            data_cfg["text_column"],
            data_cfg["sampling_rate"],
        )

elif INPUT_TYPE == "spectrogram":

    # Load TTS configuration
    with open("configs/tts_config.yaml") as f:
        tts_cfg = yaml.safe_load(f)

    # Load TTS model
    tts_model, enhancer = load_tts_model(tts_cfg, DEVICE)

    prepare_fn = lambda example: prepare_spectrogram_dataset(
        example,
        tts_model,
        enhancer,
        processor,
        data_cfg["text_column"],
        normalization=tts_cfg["tts"]["normalization"],
        enhancer_seed=tts_cfg["enhancer"]["seed"],
    )

else:
    raise ValueError(
        f"Unknown input_type: {INPUT_TYPE}"
    )


# Preprocess dataset and remove unused columns
dataset = dataset.map(
    prepare_fn,
    batched=False,
    remove_columns=dataset.column_names["train"],
)


# ============================================================
# Create data collator
# ============================================================

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=
        model.config.decoder_start_token_id,
)


# ============================================================
# Prepare training arguments
# ============================================================

OUTPUT_PATH = os.path.join(
        OUTPUT_DIR, MODEL_OUTPUT_NAME,
)


training_cfg = cfg["training"]

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_PATH,

    per_device_train_batch_size=
        training_cfg["train_batch_size"],

    per_device_eval_batch_size=
        training_cfg["eval_batch_size"],

    gradient_accumulation_steps=
        training_cfg["gradient_accumulation_steps"],

    learning_rate=
        float(training_cfg["learning_rate"]),

    warmup_steps=
        training_cfg["warmup_steps"],

    num_train_epochs=
        training_cfg["num_train_epochs"],

    gradient_checkpointing=
        training_cfg["gradient_checkpointing"],

    fp16=training_cfg["fp16"],

    optim=training_cfg["optim"],

    evaluation_strategy=
        training_cfg["evaluation_strategy"],

    save_strategy=
        training_cfg["save_strategy"],

    save_total_limit=
        training_cfg["save_total_limit"],

    logging_strategy=
        training_cfg["logging_strategy"],

    report_to=training_cfg["report_to"],

    load_best_model_at_end=
        training_cfg["load_best_model_at_end"],

    metric_for_best_model=
        training_cfg["metric_for_best_model"],

    greater_is_better=
        training_cfg["greater_is_better"],

    predict_with_generate=
        training_cfg["predict_with_generate"],

    generation_max_length=
        training_cfg["generation_max_length"],

    push_to_hub=training_cfg["push_to_hub"],

    seed=SEED,
)


# ============================================================
# Create trainer
# ============================================================

if MODEL_TYPE == "ilma":

    trainer = DummyEncoderSeq2SeqTrainer(

        args=training_args,

        model=model,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        data_collator=data_collator,

        compute_metrics=lambda pred: compute_metrics(
            pred, processor.tokenizer,
        ),

        tokenizer=processor.feature_extractor,

        dummy_input=cfg["ilma"]["dummy_input"],

        dummy_std=cfg["ilma"]["dummy_std"],

        dummy_generator=dummy_generator,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg["early_stopping_patience"]
            )
        ],
    )

else:

    trainer = Seq2SeqTrainer(

        args=training_args,

        model=model,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        data_collator=data_collator,

        compute_metrics=lambda pred:
            compute_metrics(
                pred, processor.tokenizer
            ),

        tokenizer=processor.feature_extractor,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg["early_stopping_patience"]
            )
        ],
    )


# ============================================================
# Train model
# ============================================================

trainer.train()


# ============================================================
# Save model
# ============================================================

trainer.save_model(OUTPUT_PATH)