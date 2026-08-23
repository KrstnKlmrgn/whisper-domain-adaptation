# models/lm_scorer.py
"""
Wrapper around a causal language model used for LM-based fusion.

The wrapper provides LM scoring for shallow fusion and hidden-state
extraction for Deep Fusion. LM parameters are frozen because the
language model is used as a fixed pretrained component.
"""

from typing import Optional

import torch



class LMScorer:
    """
    Wrapper for a causal LM.
    Used for:
    - shallow fusion: next token log probabilities
    - deep fusion: hidden states
    """


    def __init__(self, model, tokenizer,):

        self.model = model

        self.tokenizer = tokenizer

        self.device = model.device

        self.freeze_parameters()



    def freeze_parameters(self):

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def get_last_token_log_prob(
        self,
        input_ids: torch.LongTensor,
    ):
        """
        Returns log P(last token | previous tokens)
        input_ids:
            (batch, seq_len)
        return:
            (batch,)
        """

        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
        )

        logits = outputs.logits[:, -2, :]

        target_ids = input_ids[:, -1]

        log_probs = torch.log_softmax(
            logits, dim=-1,
        )

        token_log_prob = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)


        return token_log_prob


    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns last LM hidden layer.
        Used for deep fusion.
        """

        # If no mask is provided, derive it from the tokenizer padding ID
        if attention_mask is None:
            attention_mask = (
                input_ids != self.tokenizer.pad_token_id
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        return outputs.hidden_states[-1]