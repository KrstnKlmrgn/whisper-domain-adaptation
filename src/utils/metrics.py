import torch
import math
from evaluate import load


wer_metric = load("wer")


def compute_wer(predictions, references):
    return 100 * wer_metric.compute(
        predictions=predictions,
        references=references,
    )


def compute_ppl_old(losses):
    return torch.exp(
        torch.stack(losses).mean()
    )

def compute_ppl_test(losses):
    return torch.exp(
        torch.stack(losses).double().mean()
    )

def compute_ppl(losses):
    mean_loss = torch.stack(losses).mean().item()
    return math.exp(mean_loss)


def compute_metrics(pred, tokenizer):
    """
    Compute Word Error Rate (WER) during evaluation.
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = (
        tokenizer.pad_token_id
    )

    # We do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )

    label_str = tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True
    )

    wer = compute_wer(pred_str, label_str)

    return {"wer": wer}



def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """

    pred_ids = torch.argmax(
        logits,
        #logits[0]
        dim=-1
    )

    return pred_ids