"""
Whisper with hidden-state-level Deep Fusion.

This module extends Whisper by combining its decoder hidden states
with hidden states from an external causal language model. The LM
representations are optionally gated and concatenated with the
Whisper representations before the final vocabulary projection.

The language model is kept frozen during training.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right


class DeepFusionWhisper(WhisperForConditionalGeneration):
    """
    Whisper model extended with GPT-based language model fusion.

    The model combines Whisper decoder hidden states with hidden states
    from an external language model. A learned gate controls the influence
    of the language model before projecting to vocabulary logits.
    """

    def __init__(
        self,
        pretrained_whisper,
        lm_scorer,
        use_lm=True,
        copy_weights=False,
    ):

        self.use_lm = use_lm

        self.config = pretrained_whisper.config
        self.generation_config = pretrained_whisper.generation_config

        super().__init__(self.config)

        self.lm_scorer = lm_scorer
        self.model = pretrained_whisper.model

        # Get hidden state dimension for LM and Whisper
        lm_hidden_dim = lm_scorer.model.lm_head.in_features
        whisper_hidden_dim = self.config.d_model

        if use_lm:
            output_dim = whisper_hidden_dim + lm_hidden_dim
        else:
            output_dim = whisper_hidden_dim


        self.proj_out = nn.Linear(
            in_features=output_dim,
            out_features=self.config.vocab_size,
            bias=False,
        )

        # The gate controls how strongly the LM representation contributes
        # to the fused representation
        self.gate_type = gate_type

        if gate_type == "coarse":

            # One gate value per token, shared across all hidden dimensions
            self.lm_gate = nn.Linear(
                in_features=lm_hidden_dim,
                out_features=1,
                bias=True,
            )

        elif gate_type == "fine":

            # One gate value for each hidden dimension of each token
            self.lm_gate = nn.Linear(
                in_features=lm_hidden_dim,
                out_features=lm_hidden_dim,
                bias=True,
            )

        else:
            raise ValueError(
                f"Unknown gate_type: {gate_type}. "
                f"Expected 'coarse' or 'fine'."
            )

        self.sigmoid = nn.Sigmoid()

        self.max_target_positions = self.config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()

        # Copy weights from old/pre-trained output layer
        if copy_weights:
            # Copy whisper weights
            whisper_weights = (
                pretrained_whisper.get_output_embeddings()
                .weight.clone().detach()
            )
            self.proj_out.weight.data[:, :whisper_hidden_dim] = whisper_weights.data

            if use_lm:
                # Copy lm weights
                lm_weights = (
                    lm_scorer.model.transformer.get_input_embeddings()
                    .weight.clone().detach()
                )
                self.proj_out.weight.data[:, whisper_hidden_dim:] = lm_weights.data


    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels length {labels.shape[1]} exceeds "
                    f"maximum {self.max_target_positions}"
                )

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    self.config.pad_token_id,
                    self.config.decoder_start_token_id,
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        whisper_hidden_states = outputs.last_hidden_state

        if self.use_lm:
            lm_hidden_states = self.lm_scorer.get_hidden_states(
                decoder_input_ids,
                #use_cache=use_cache
            )

            # Compute a sigmoid gate and apply it to the LM hidden states.
            # Broadcasting handles the coarse-grained case where the gate has
            # shape (batch, sequence_length, 1)
            gate = self.sigmoid(self.lm_gate(lm_hidden_states))

            gated_lm_hidden_states = gate * lm_hidden_states

            # Fuse Whisper and LM representations at the hidden-state level
            hidden_states = torch.cat(
                (whisper_hidden_states, gated_lm_hidden_states), dim=-1,
            )

        else:
            hidden_states = whisper_hidden_states

        logits = self.proj_out(hidden_states)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # Move labels to correct device to enable PP
            labels = labels.to(logits.device)

            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.reshape(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (
                (loss,) + output
                if loss is not None
                else output
            )


        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    '''
    def freeze_encoder_parameters(self):

        encoder = self.model.get_encoder()

        if encoder is not None:
            encoder._freeze_parameters()
            encoder.gradient_checkpointing = False


    def freeze_decoder_parameters(self):

        decoder = self.model.get_decoder()

        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad = False

            decoder._requires_grad = False
            decoder.gradient_checkpointing = False
    #'''


    def init_weights(self):
        """
        Initialize weights without tying input embeddings
        to the output projection.
        """

        self.apply(self._initialize_weights)
        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        #self.tie_weights()
