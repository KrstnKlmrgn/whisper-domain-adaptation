import torch
from torch import nn

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    WhisperTokenizer,
)


# ============================================================
# Tokenizer utils
# ============================================================

def load_lm_tokenizer(
    tokenizer_type,
    lm_name,
    whisper_model,
    whisper_path,
    language=None,
    task=None,
):
    """
    Load tokenizer used for GPT-2 LM evaluation/fine-tuning.
    """

    if tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(
            lm_name
        )

    elif tokenizer_type == "whisper_en":
        tokenizer = WhisperTokenizer.from_pretrained(
            whisper_path + whisper_model,
            no_timestamps=True,
        )

    elif tokenizer_type == "whisper_multi":
        tokenizer = WhisperTokenizer.from_pretrained(
            whisper_path + whisper_model,
            language=language,
            task=task,
            no_timestamps=True,
        )

    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}"
        )

    # GPT-2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer



# ============================================================
# Model
# ============================================================

def load_lm_model(
    model_name,
    model_path,
    tokenizer,
    device,
):
    """
    Load GPT-2 causal LM and adapt vocabulary/config
    to tokenizer.
    """

    model = GPT2LMHeadModel.from_pretrained(
        model_path + model_name
    )

    # Resize embeddings for Whisper vocabulary
    if model.transformer.wte.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    model.to(device)


    return model


# ============================================================
# Embedding transfer (optional experiment)
# ============================================================


def _initialize_with_fvt(
    token,
    gpt2_tokenizer,
    old_embeddings,
):
    """
    Initialize a Whisper-only token using the average
    embedding of its GPT-2 subtokens.
    """

    adjusted_token = (
        " " + token[1:]
        if token.startswith("Ġ")
        else token
    )

    subtoken_ids = gpt2_tokenizer(
        adjusted_token,
        add_special_tokens=False,
    ).input_ids

    if len(subtoken_ids) == 0:
        return None

    return old_embeddings[subtoken_ids].mean(dim=0)


def reorder_embeddings(
    model,
    whisper_tokenizer,
    model_name,
    reorder_type="copy",
):
    """
    Reorder GPT-2 embeddings to match the Whisper vocabulary.

    reorder_type:
        "copy" : copy shared token embeddings.
        "fvt"  : copy shared tokens and initialize
                 Whisper-only tokens with FVT.
    """

    if reorder_type not in {"copy", "fvt"}:
        raise ValueError(
            f"Unknown reorder type: {reorder_type}"
        )

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        model_name
    )

    gpt2_vocab = gpt2_tokenizer.get_vocab()
    whisper_vocab = whisper_tokenizer.get_vocab()

    old_embeddings = (
        model.transformer.wte.weight
        .detach()
        .clone()
    )

    new_embeddings = old_embeddings.clone()

    shared = 0
    initialized = 0

    for token, whisper_id in whisper_vocab.items():

        gpt2_id = gpt2_vocab.get(token)

        if gpt2_id is not None:
            new_embeddings[whisper_id] = (
                old_embeddings[gpt2_id]
            )
            shared += 1

        elif reorder_type == "fvt":
            embedding = _initialize_with_fvt(
                token,
                gpt2_tokenizer,
                old_embeddings,
            )

            if embedding is not None:
                new_embeddings[whisper_id] = embedding
                initialized += 1

    model.transformer.wte.weight.data.copy_(new_embeddings)

    model.tie_weights()

    print(f"Shared tokens copied: {shared}")

    if reorder_type == "fvt":
        print(f"FVT initialized: {initialized}")

    return model


