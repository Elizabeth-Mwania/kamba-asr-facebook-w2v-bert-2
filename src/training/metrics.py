# src/training/metrics.py
from typing import Dict, List
import numpy as np
import torch
import os
import json
import evaluate
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor
from transformers.trainer_utils import PredictionOutput

from src.data.dataset import ASRProcessor

import logging

from post_processing.normalization import KambaNormalizer
_kamba_normalizer = KambaNormalizer()

def preprocess_logits_for_metrics(logits: torch.Tensor, 
                                  labels: torch.Tensor) -> torch.Tensor:
    """
    Preprocess logits for metrics.
    Args:
        logits: Logits tensor of shape (batch_size, sequence_length, vocab_size)
        labels: Labels tensor of shape (batch_size, sequence_length)
    Returns:
        Predicted token ids of shape (batch_size, sequence_length)
    """
    # only keep predicted token ids, not full logit tensor
    return logits.argmax(dim=-1)



class ASRMetrics:
    def __init__(self,
                 processor: ASRProcessor,
                 wer_metric: evaluate.Metric,
                 cer_metric: evaluate.Metric,
                 output_dir: str = None):
        self.processor = processor
        self.wer_metric = wer_metric
        self.cer_metric = cer_metric
        self.output_dir = output_dir
        self._call_count = 0

    def compute_metrics(self, pred: PredictionOutput) -> Dict[str, float]:
        k = self._call_count
        self._call_count += 1

        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        pred_str  = [_kamba_normalizer.normalize(p) for p in pred_str]
        label_str = [_kamba_normalizer.normalize(r) for r in label_str]

        if self.output_dir is not None:
            # creare output directory if it does not exist
            os.makedirs(self.output_dir, exist_ok=True)

            predictions_and_references = {
                f"ID_{i}": {"prediction": pred_str[i], "reference": label_str[i]}
                for i in range(len(pred_str))
            }
            json_path = os.path.join(self.output_dir, f"predictions_{k}.json")
            with open(json_path, "w" , encoding='utf-8') as f:
                json.dump(predictions_and_references, f, ensure_ascii=False, indent=4)
            logging.info(f"predictions and references saved to {json_path}")

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        for i in range(min(10, len(pred_str))):
            sample_wer = _simple_wer(pred_str[i], label_str[i])
            sample_cer = _simple_cer(pred_str[i], label_str[i])
            print(f"Sample {i}:")
            print(f"Prediction: {pred_str[i]}")
            print(f"Reference: {label_str[i]}")
            print(f"WER: {sample_wer*100:.4f}%")
            print(f"CER: {sample_cer*100:.4f}%")
            print("-"*75)

        combined_error = (0.5 * wer) + (0.5 * cer)
        score = (1 - combined_error) * 100

        return {"wer": wer, "cer": cer, "score": score}


# NOTE: this code was added to be used as WER/CER metrics on individual samples 
# which were shown during training to debug the model 
# using Hugging Face metrics library for individual samples was not working as expected
def _simple_wer(pred: str, ref: str) -> float:
    # simple wer calculation for debug purposes
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = _levenshtein(pred_words, ref_words)
    return distance / len(ref_words)

def _simple_cer(pred: str, ref: str) -> float:
    # simple cer calculation for debug purposes
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    distance = _levenshtein(list(pred), list(ref))
    return distance / len(ref)

def _levenshtein(a: list, b: list) -> int:
    # levenshtein distance
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = range(len(b) + 1)
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]