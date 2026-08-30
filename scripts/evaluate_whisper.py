#!/usr/bin/env python3
"""Compute WER and CER for a saved Whisper checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import List

import evaluate
import torch
from datasets import Audio, load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor


ALLOWED = frozenset("abcdefghijklmnoprstuvwyĩũ' 0123456789")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def normalize_kamba(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower().translate(APOSTROPHES)
    return re.sub(r"\s+", " ", "".join(char if char in ALLOWED else " " for char in text)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Whisper checkpoint directory, e.g. checkpoint-1000")
    parser.add_argument("--data_dir", default="data/kamba_10h_v1")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--language", default="sw")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--metrics_path", default=None, help="Optional path for a small JSON metrics file.")
    parser.add_argument("--predictions_path", default=None, help="Optional path for predictions JSON.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for practical Whisper evaluation.")
    print(f"Evaluating on {torch.cuda.get_device_name(0)}")

    data = load_from_disk(str(Path(args.data_dir) / "dataset"))
    dataset = data[args.split].cast_column("audio", Audio(sampling_rate=16_000))
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint).to(device).eval()
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    predictions: List[str] = []
    references: List[str] = []
    for start in range(0, len(dataset), args.batch_size):
        examples = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
        features = processor.feature_extractor(
            [row["audio"]["array"] for row in examples], sampling_rate=16_000, return_tensors="pt"
        )
        with torch.inference_mode():
            generated_ids = model.generate(features.input_features.to(device), max_length=225)
        predictions.extend(normalize_kamba(text) for text in processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        references.extend(normalize_kamba(row["transcripts"]) for row in examples)

    valid = [(hyp, ref) for hyp, ref in zip(predictions, references) if ref]
    hypotheses, refs = zip(*valid)
    metrics = {
        "model_checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "n_samples": len(valid),
        "wer": evaluate.load("wer").compute(predictions=list(hypotheses), references=list(refs)),
        "cer": evaluate.load("cer").compute(predictions=list(hypotheses), references=list(refs)),
    }
    print(json.dumps(metrics, indent=2))

    if args.metrics_path:
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.predictions_path:
        Path(args.predictions_path).write_text(
            json.dumps([{"reference": ref, "prediction": hyp} for hyp, ref in zip(references, predictions)], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
