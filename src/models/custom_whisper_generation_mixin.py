"""
Custom generation mixin utilities for Whisper with shallow language model fusion.

This module extends Hugging Face's Whisper generation logic to incorporate
scores from an external language model during beam search.
"""

import copy
import inspect
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperNoSpeechDetection,
    WhisperTimeStampLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.generation_whisper import (
    WhisperGenerationMixin,
    _get_attr_from_logit_processors,
    _pad_to_max_length,
)
from transformers.models.whisper.tokenization_whisper import (
    TASK_IDS,
    TO_LANGUAGE_CODE,
)
from transformers.utils import logging

from custom_beam_search_scorer import CustomBeamSearchScorer



class CustomWhisperGenerationMixin(WhisperGenerationMixin):
    """
    Extends Whisper generation with shallow language model fusion.

    Hugging Face's Whisper generation logic is reused and modified where
    language model scores need to be incorporated into beam search.
    """

    def __init__(
        self,
        lm_scorer=None,
        lm_weight=0.1,
        tokenizer=None,
    ):
        super().__init__()

        self.lm_scorer = lm_scorer
        self.lm_weight = lm_weight
        self.tokenizer = tokenizer


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cached key/value states according to the selected beams.
        Copied from Hugging Face Transformers v4.45.2:
        ``transformers.models.whisper.modeling_whisper``.
        """

        reordered_past = ()

        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )

        return reordered_past



    def generate_with_fallback(self, segment_input, decoder_input_ids,
        cur_bsz, batch_idx_map, seek, num_segment_frames, max_frames,
        temperatures, generation_config, logits_processor, stopping_criteria,
        prefix_allowed_tokens_fn, synced_gpus,
        return_token_timestamps, do_condition_on_prev_tokens,
        is_shortform, batch_size, attention_mask, kwargs,):

        """
        Adapted from Hugging Face Transformers v4.45.2:
        ``transformers.models.whisper.generation_whisper.WhisperGenerationMixin``.

        The original beam search call is replaced with ``call_beam_search``
        to enable shallow fusion with an external language model.
        """

        # ============================================================
        # The function starts with the original Hugging Face 
        # implementation.
        # ============================================================

        kwargs = copy.copy(kwargs)

        # 6.6 Batch generate current chunk
        seek_sequence_list = [None for _ in range(cur_bsz)]
        seek_outputs_list = [None for _ in range(cur_bsz)]
        needs_fallback = [False for _ in range(cur_bsz)]
        should_skip = [False for _ in range(cur_bsz)]
        fallback_index_map = list(range(cur_bsz))
        if generation_config.no_speech_threshold is not None:
            self._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)

        for fallback_idx, temperature in enumerate(temperatures):
            generation_config.do_sample = temperature is not None and temperature > 0.0
            generation_config.temperature = temperature if generation_config.do_sample else 1.0
            if generation_config.do_sample:
                generation_config.num_beams = 1

            generate_kwargs = copy.copy(kwargs)
            for key in ["do_sample", "temperature", "num_beams"]:
                if key in generate_kwargs:
                    del generate_kwargs[key]

            cur_bsz = decoder_input_ids.shape[0]
            if generation_config.cache_implementation == "static" and cur_bsz < batch_size:
                segment_input = F.pad(segment_input, (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0)
                decoder_input_ids = F.pad(
                    decoder_input_ids, (0, 0, 0, batch_size - cur_bsz), value=generation_config.pad_token_id
                )
                if generate_kwargs.get("decoder_attention_mask") is not None:
                    generate_kwargs["decoder_attention_mask"] = F.pad(
                        generate_kwargs["decoder_attention_mask"], (0, 0, 0, batch_size - cur_bsz), value=True
                    )
                if generate_kwargs.get("encoder_outputs") is not None:
                    generate_kwargs["encoder_outputs"] = F.pad(
                        generate_kwargs["encoder_outputs"], (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0
                    )


            # ============================================================
            # MODIFIED PART:
            # Use custom beam search to incorporate LM scores.
            # ============================================================

            seek_outputs = self.call_beam_search(
                segment_input,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )


            # ============================================================
            # From here on, the original Hugging Face implementation
            # continues.
            # ============================================================

            model_output_type = type(seek_outputs)

            # post-process sequence tokens and outputs to be in list form
            seek_sequences, seek_outputs = self._postprocess_outputs(
                seek_outputs=seek_outputs,
                decoder_input_ids=decoder_input_ids,
                return_token_timestamps=return_token_timestamps,
                generation_config=generation_config,
                is_shortform=is_shortform,
            )
            if cur_bsz < batch_size:
                seek_sequences = seek_sequences[:cur_bsz]
                seek_outputs = seek_outputs[:cur_bsz]

            # 6.7 Extract cut sequences from every sequence and check if fallback should be applied
            # Loop over each decoded audio individually as each decoding can be of a different length
            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            for i, seek_sequence in enumerate(seek_sequences):
                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                prev_i = batch_idx_map[fallback_index_map[i]]
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]

                # remove eos token id
                if is_not_final and seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]
                    if return_token_timestamps and not is_shortform:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-1]

                # remove all padding tokens
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]
                    if return_token_timestamps and not is_shortform:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-num_paddings]

                # check which sequences in batch need fallback & which should be skipped
                needs_fallback[i], should_skip[i] = self._need_fallback(
                    seek_sequence,
                    seek_outputs,
                    i,
                    logits_processor,
                    generation_config,
                    self.config.vocab_size,
                    temperature,
                )

                seek_sequence_list[fallback_index_map[i]] = seek_sequence
                seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
                is_low_temperature = temperature is None or temperature < 0.5
                do_condition_on_prev_tokens[fallback_index_map[i]] = (
                    generation_config.condition_on_prev_tokens and is_low_temperature
                )

                if needs_fallback[i]:
                    new_fallback_index_map.append(fallback_index_map[i])
                    new_segment_input.append(segment_input[i])
                    new_decoder_input_ids.append(decoder_input_ids[i])
                    if "decoder_attention_mask" in kwargs:
                        new_decoder_attention_mask.append(kwargs["decoder_attention_mask"][i])

            fallback_index_map = new_fallback_index_map

            # if no sequence needs to be run with temperature fallback, we're finished
            if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
                seek_sequences = seek_sequence_list
                seek_outputs = seek_outputs_list
                break

            # if we're still in the loop, make sure that decoder_input_ids and segment inputs are tensors
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            segment_input = torch.stack(new_segment_input)
            if "decoder_attention_mask" in kwargs:
                kwargs["decoder_attention_mask"] = torch.stack(new_decoder_attention_mask)

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type


    def call_beam_search(self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:

        """
        Prepare generation inputs and run beam search with a custom scorer.

        This function is adapted from the general Hugging Face
        ``GenerationMixin.generate`` implementation. It reproduces the
        relevant generation setup steps because the standard ``generate``
        interface does not provide a way to pass a custom beam search scorer.
        """

        # ============================================================
        # The function starts with relevant parts the original
        # Hugging Face  implementation.
        # ============================================================

        # Handle generation_config and kwargs that might update it, and validate the .generate() call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        synced_gpus = False

        # Prepare model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs)

        # Determine batch size and device
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device

        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )
        # Prepare input_ids for generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,)

        # 6. Prepare max_length depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,)

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
        user_defined_cache = model_kwargs.get(cache_name)
        max_cache_length = generation_config.max_length

        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder):

            max_cache_length += inputs_tensor.shape[1]
            self._prepare_cache_for_generation(
            generation_config, model_kwargs, None, batch_size, max_cache_length, device)

        # Prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs
        )

        # Prepare beam search scorer

        # ============================================================
        # MODIFIED PART:
        # Create the custom beam search scorer with LM integration.
        #
        # The standard Hugging Face generation code creates its own
        # beam search scorer internally. Here, the custom scorer is
        # constructed and passed
        # ============================================================

        beam_scorer = CustomBeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
            lm_scorer=self.lm_scorer,
            lm_weight=self.lm_weight,
        )


        # ============================================================
        # From here on, the original Hugging Face beam search
        # execution is used with the custom scorer.
        # ============================================================

        # Interleave input_ids with 'num_beams' additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


        return result

