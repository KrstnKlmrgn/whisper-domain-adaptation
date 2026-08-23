"""
This file contains a modified version of HuggingFace's
Seq2SeqTrainer.

Original implementation:
transformers.Seq2SeqTrainer

Modifications:
- Replace Whisper encoder outputs with artificial hidden states.
- Enable decoder-only language model adaptation without audio input.

All non-modified training behavior is inherited from
the original HuggingFace implementation.
"""


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import BaseModelOutput

from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
)

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from transformers.trainer import _is_peft_model


class DummyEncoderSeq2SeqTrainer(Seq2SeqTrainer):

    """
    Seq2SeqTrainer for Whisper internal LM adaptation.

    The Whisper encoder is bypassed by replacing its hidden
    states with fixed dummy representations. This allows training
    the decoder language model without using acoustic information
    and enables internal decoder LM adaptation.

    The modified parts are:
        1. Creation of artificial encoder outputs.
        2. Injection of these outputs during training loss computation.
        3. Injection of these outputs during generation/evaluation.

    All remaining training logic follows the original
    HuggingFace Seq2SeqTrainer implementation.
    """


    def __init__(
        self,
        *args,
        dummy_input="zero",
        dummy_std=None,
        dummy_generator=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dummy_input = dummy_input
        self.dummy_std = dummy_std
        self.dummy_generator = dummy_generator


    def create_dummy_encoder_outputs(
        self, batch_size, max_pos, model_dim,
    ):
        """
        Create artificial encoder hidden states.
        """

        device = self.model.device

        if self.dummy_input == "zero":

            hidden_states = torch.zeros(
                size=(batch_size, max_pos, model_dim),
                device=device,
            )
        elif self.dummy_input == "random":

            if self.dummy_generator is None:
                raise ValueError(
                    "dummy_generator required for random input"
                )

            hidden_states = torch.normal(
                mean=0.0,
                std=self.dummy_std,
                size=(batch_size, max_pos, model_dim),
                generator=self.dummy_generator,
                device=device,
            )

        else:
            raise ValueError(
                f"Unknown dummy_input: {self.dummy_input}"
            )

        return BaseModelOutput(last_hidden_state=hidden_states)



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss using dummy encoder outputs.

        The Whisper encoder output is replaced with dummy hidden states
        before the model computes the loss.
        """

        # ============================================================
        # MODIFIED PART:
        # Replace Whisper encoder output with dummy hidden states.
        # ============================================================

        encoder_outputs = self.create_dummy_encoder_outputs(
            batch_size=inputs["input_features"].shape[0],
            max_pos=model.config.max_source_positions,
            model_dim=model.config.d_model,
        )
        outputs = model(
            **inputs, encoder_outputs=encoder_outputs,
        )


        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step.

        Adapted from the Hugging Face Seq2SeqTrainer implementation.

        Modification:
        Replace the Whisper encoder output with artificial encoder
        representations during generation and loss computation.
        """

        # ============================================================
        # The function starts with the original Hugging Face
        # implementation.
        # ============================================================

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }


        # ============================================================
        # MODIFIED PART:
        # During generation, provide artificial encoder
        # representations instead of running the Whisper encoder.
        # ============================================================

        encoder_outputs = self.create_dummy_encoder_outputs(
            batch_size=inputs["input_features"].shape[0],
            max_pos=model.config.max_source_positions,
            model_dim=model.config.d_model,
        )

        generated_tokens = self.model.generate(
            **generation_inputs,
            encoder_outputs=encoder_outputs,
            **gen_kwargs,
        )


        # Temporary compatibility workaround from the original
        # Hugging Face Seq2SeqTrainer implementation
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config

        # In case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():

                    # ========================================================
                    # MODIFIED PART 2:
                    # Provide artificial encoder representations instead
                    # of running the Whisper encoder.
                    # ========================================================
                    outputs = model(**inputs, encoder_outputs=encoder_outputs)


                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels