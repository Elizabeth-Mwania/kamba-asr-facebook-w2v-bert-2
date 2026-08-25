#!/usr/bin/env python3
"""Fine-tune Whisper on the reusable Kamba benchmark without feature caching."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
from datasets import Audio, DatasetDict, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


ALLOWED = frozenset("abcdefghijklmnoprstuvwyĩũ' 0123456789")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def normalize_kamba(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower().translate(APOSTROPHES)
    text = "".join(char if char in ALLOWED else " " for char in text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class WhisperCollator:
    """Decode audio and create Whisper features only for the current batch."""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [item["audio"]["array"] for item in features]
        sampling_rates = {item["audio"]["sampling_rate"] for item in features}
        if len(sampling_rates) != 1:
            raise ValueError("A batch has mixed audio sampling rates.")
        inputs = self.processor.feature_extractor(
            audio_arrays, sampling_rate=sampling_rates.pop(), return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            [normalize_kamba(item["transcripts"]) for item in features],
            padding=True,
            return_tensors="pt",
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (label_ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            label_ids = label_ids[:, 1:]
        return {"input_features": inputs.input_features, "labels": label_ids}


def build_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute(prediction) -> Dict[str, float]:
        labels = prediction.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predictions = [normalize_kamba(x) for x in processor.tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)]
        references = [normalize_kamba(x) for x in processor.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        valid = [(p, r) for p, r in zip(predictions, references) if r]
        if not valid:
            return {"wer": 1.0, "cer": 1.0}
        hyp, ref = zip(*valid)
        return {
            "wer": wer_metric.compute(predictions=list(hyp), references=list(ref)),
            "cer": cer_metric.compute(predictions=list(hyp), references=list(ref)),
        }

    return compute


def evaluate_and_save(
    trainer: Seq2SeqTrainer, processor: WhisperProcessor, dataset, output_dir: Path, split: str
) -> Dict[str, float]:
    metrics = trainer.evaluate(dataset, metric_key_prefix=split)
    prediction = trainer.predict(dataset, metric_key_prefix=split)
    references = prediction.label_ids.copy()
    references[references == -100] = processor.tokenizer.pad_token_id
    hypotheses = [normalize_kamba(x) for x in processor.tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)]
    references = [normalize_kamba(x) for x in processor.tokenizer.batch_decode(references, skip_special_tokens=True)]
    (output_dir / f"predictions_{split}.json").write_text(
        json.dumps([{"reference": ref, "prediction": hyp} for hyp, ref in zip(hypotheses, references)], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        f"{split}_wer": float(metrics[f"{split}_wer"]),
        f"{split}_cer": float(metrics[f"{split}_cer"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/kamba_10h_v1")
    parser.add_argument("--output_dir", default="outputs/kamba-whisper-small-10h")
    parser.add_argument("--model_id", default="openai/whisper-small")
    parser.add_argument("--language", default="sw", help="Whisper prompt language; sw is a documented proxy for Kamba.")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable only when needed for a smaller GPU; it is off by default for stable T4 training.",
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    data_path = Path(args.data_dir) / "dataset"
    if not (data_path / "dataset_dict.json").exists():
        raise FileNotFoundError(f"No materialized benchmark at {data_path}. Run materialize_kamba_subset.py first.")
    data: DatasetDict = load_from_disk(str(data_path))
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = not args.gradient_checkpointing

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=25,
        dataloader_num_workers=2,
        seed=args.seed,
    )
    # Transformers renamed this argument in newer releases. Support the local
    # pinned version and current Colab wheels without forcing a source build.
    if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=WhisperCollator(processor),
        compute_metrics=build_metrics(processor),
    )
    if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    checkpoint = args.resume_from_checkpoint or get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)

    model_dir = output_dir / "best_model"
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    results = {
        "model_id": args.model_id,
        "dataset_manifest": str(Path(args.data_dir) / "benchmark_manifest.json"),
        "train_clips": len(data["train"]),
        "validation_clips": len(data["validation"]),
        "test_clips": len(data["test"]),
        **evaluate_and_save(trainer, processor, data["validation"], output_dir, "validation"),
        **evaluate_and_save(trainer, processor, data["test"], output_dir, "test"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Finished. WER/CER are in %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
