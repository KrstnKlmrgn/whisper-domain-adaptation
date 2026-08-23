import torch

from transformers import WhisperProcessor


def load_whisper_processor(
    whisper_path,
    whisper_model,
    language=None,
    task=None,
):

    if whisper_model.endswith("en"):

        processor = WhisperProcessor.from_pretrained(
            whisper_path + whisper_model,
            no_timestamps=True,
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            no_timestamps=True
        )

    else:

        processor = WhisperProcessor.from_pretrained(
            whisper_path + whisper_model,
            language=language,
            task=task,
            no_timestamps=True,
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=task,
            no_timestamps=True,
        )

    return processor, forced_decoder_ids


def configure_whisper_generation(
    model,
    whisper_model,
    language,
    task,
    forced_decoder_ids=None,
    max_length=None,
    num_beams=None,
    return_timestamps=None,
    renormalize_logits=True,
):
    """
    Configure Whisper generation settings.
    """

    if not whisper_model.endswith("en"):
        model.generation_config.language = language
        model.generation_config.task = task

    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.renormalize_logits = renormalize_logits

    if max_length is not None:
        model.generation_config.max_length = max_length

    if num_beams is not None:
        model.generation_config.num_beams = num_beams
        model.generation_config.early_stopping = num_beams > 1

    if return_timestamps is not None:
        model.generation_config.return_timestamps = return_timestamps


# ------------------------------------------------------------
# Freezing Parts of Whisper Model
# ------------------------------------------------------------

def freeze_encoder(model):

    """
    Freeze encoder transformer blocks
    """

    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


def freeze_decoder(model):

    """
    Freeze decoder transformer blocks
    """

    decoder = model.get_decoder()

    for param in decoder.parameters():
        param.requires_grad = False

    decoder._requires_grad = False
    decoder.gradient_checkpointing = False


def freeze_cross_attention(model):

    """
    Freeze cross-attention layers in decoder
    """

    decoder = model.get_decoder()

    for layer in decoder.layers:
        for param in layer.encoder_attn.parameters():
            param.requires_grad = False

        for param in layer.encoder_attn_layer_norm.parameters():
            param.requires_grad = False


def freeze_decoder_except_embeddings(model):

    """
    Freeze decoder transformer blocks while keeping the tied
.   input/output token embedding matrix trainable.
    """

    decoder = model.get_decoder()

    for param in decoder.parameters():
        param.requires_grad = False

    decoder.gradient_checkpointing = False

    # Tied input/output embedding
    model.proj_out.weight.requires_grad = True