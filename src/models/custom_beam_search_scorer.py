"""
Custom beam search scorer for shallow fusion.

Extends Hugging Face's BeamSearchScorer by incorporating
language model scores into beam search candidate selection.
"""

from collections import UserDict
from typing import Dict, List, Optional, Union

import torch
from transformers import BeamSearchScorer


class CustomBeamSearchScorer(BeamSearchScorer):
    """
    Beam search scorer with language model score integration.

    Extends Hugging Face's BeamSearchScorer by adding language
    model scores to the beam search candidates before the original
    beam search processing is performed.
    """

    def __init__(
        self,
        *args,
        lm_scorer=None,
        lm_weight=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lm_scorer = lm_scorer
        self.lm_weight = lm_weight


    def add_lm_scores(
        self, input_ids, next_tokens, next_scores, next_indices,
    ):
        """
        Adds language model scores to the beam search scores.
        """

        scores_with_lm = torch.zeros_like(next_scores)
        batch_size, num_tokens = next_scores.shape

        for b in range(batch_size):

            for i in range(num_tokens):

                beam_index = next_indices[b, i]

                hypothesis = torch.cat(
                    [
                        input_ids[beam_index],
                        next_tokens[b, i].unsqueeze(0)
                    ],
                    dim=0
                )
                lm_score = self.lm_scorer.get_last_token_log_prob(
                    hypothesis.unsqueeze(0)
                )
                scores_with_lm[b, i] = (
                    next_scores[b, i] + self.lm_weight * lm_score
                )

        return scores_with_lm


    def rerank_scores(self, next_tokens, next_scores, next_indices):
        """
        Sorts candidate tokens by score and reorder their corresponding
        beam indices accordingly.
        """

        sorted_next_scores, indices = next_scores.sort(
            dim=-1, descending=True, stable=True
        )
        sorted_next_tokens = next_tokens.gather(
            dim=1, index=indices
        )
        sorted_next_indices = next_indices.gather(
            dim=1, index=indices
        )

        return (sorted_next_tokens, sorted_next_scores, sorted_next_indices)


    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[Union[int, torch.Tensor]] = None,
        eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
        ) -> Dict[str, torch.Tensor]:

        """
        Process candidate tokens and update the beam search state.

        Modified from Hugging Face Transformers v4.45.2:
        ``transformers.generation.beam_search.BeamSearchScorer.process``.

        Modifications:
            - incorporate language model scores and
            - rerank next tokens
            before selecting the next beam candidates.
        """

        # ============================================================
        # The function starts with the original Hugging Face
        # implementation.
        # ============================================================

        # add up to the length which the next_scores is calculated on (including decoder prompt)
        cur_len = input_ids.shape[-1] + 1
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer.")
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer.")

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue


            # ========================================================
            # MODIFIED PART:
            # Add language model scores and reorder the candidate
            # tokens before continuing with the original beam search.
            # ========================================================

            # Add lm scores to next_scores
            scores_with_lm = self.add_lm_scores(
                input_ids,next_tokens, next_scores, next_indices
            )
            # Rerank scores
            next_tokens, next_scores, next_indices = self.rerank_scores(
                next_tokens, scores_with_lm, next_indices
            )


            # ========================================================
            # From here on, the original Hugging Face implementation
            # continues.
            # ========================================================

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None
                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                        generated_len=cur_len - decoder_prompt_len,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break
            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )
            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
            )
        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),})

    def process_alt(
        self,
        input_ids,
        next_scores,
        next_tokens,
        next_indices,
        **kwargs,
    ):
        """
        Apply language model scores to beam search candidates and
        delegate the remaining beam search processing to Hugging Face.
        """

        scores_with_lm = self.add_lm_scores(
            input_ids, next_tokens, next_scores, next_indices,
        )

        next_tokens, next_scores, next_indices = self.rerank_scores(
            next_tokens, scores_with_lm, next_indices,
        )

        return super().process(
            input_ids, next_scores, next_tokens, next_indices, **kwargs,
        )