#!/usr/bin/env python3
"""
Kamba ASR Evaluation Script
Adapted from evaluate_model_mono_asr_lid_batch.py (Ethio-ASR)

Evaluates a fine-tuned Wav2Vec2-BERT model on the Kamba test split.
Loads data from HuggingFace streaming (no full download needed).

Usage:
    python scripts/evaluate_kamba_asr.py \
        --model_path outputs/kamba-asr-facebook-w2v-bert-2/<experiment_name> \
        --experiment_name kamba_w2v_bert_10h \
        --split test \
        --batch_size 4
"""

import os
import sys
import re
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple

warnings.filterwarnings("ignore")

# ── Environment setup (mirrors train_model.py) ────────────────────────────────
os.environ['NUMBA_CACHE_DIR'] = '/tmp/'
os.environ['NUMBA_DISABLE_JIT'] = '1'

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Load .env
env_path = project_root / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip("'").strip('"')
                except ValueError:
                    continue


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


# ── Imports (after path setup) ────────────────────────────────────────────────
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Audio, Dataset
from transformers import AutoProcessor, AutoModelForCTC
import evaluate

# import post_processing.normalization as norm_module
import post_processing.normalization 

# KambaNormalizer()
kamba_normalizer = post_processing.normalization.KambaNormalizer()

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
# DDD-Kenya dataset transcript column
TRANSCRIPT_COL = "transcripts"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load fine-tuned model and processor from local path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    logging.info(f"Loading model from: {model_path}")
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    model.eval()
    logging.info(f"Model loaded on device: {device}")
    return model, processor, device


# ── Data loading ──────────────────────────────────────────────────────────────
def load_test_split(test_sample_size: int = 500, seed: int = 42) -> Dataset:
    """
    Stream the DDD-Kenya Kamba dataset and collect the test split.
    Mirrors the same logic used in dataset.py during training —
    first 500 clips = test, next 500 = val, rest = train.
    """
    logging.info("Streaming DDD-Kenya/Kamba-ASR-Data-Subset-484H for test split...")
    stream = load_dataset(
        "DDD-Kenya/Kamba-ASR-Data-Subset-484H",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    test_rows = []
    for i, example in enumerate(stream):
        if len(test_rows) >= test_sample_size:
            break
        test_rows.append(example)

    logging.info(f"Collected {len(test_rows)} test clips")
    ds = Dataset.from_list(test_rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_batch(
    audio_arrays: List,
    model: AutoModelForCTC,
    processor: AutoProcessor,
    device: torch.device,
) -> List[str]:
    """Transcribe a batch of audio arrays. Returns raw decoded strings."""
    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_kamba(
    dataset: Dataset,
    model: AutoModelForCTC,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int = 4,
    n_samples: int = None,
) -> Dict[str, Any]:
    """
    Evaluate the model on a Kamba dataset split.
    Returns WER, CER, and sample-level predictions.
    """
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    logging.info(f"Evaluating on {len(dataset)} samples...")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions, references = [], []

    for batch_start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Transcribing"
    ):
        batch = dataset.select(
            range(batch_start, min(batch_start + batch_size, len(dataset)))
        )
        audio_arrays = [ex["audio"]["array"] for ex in batch]
        raw_preds = transcribe_batch(audio_arrays, model, processor, device)

        for ex, pred in zip(batch, raw_preds):
            # Normalise prediction using KambaNormalizer instance
            pred_text = kamba_normalizer.normalize(pred)

            # Get reference — handle both column name variants
            if TRANSCRIPT_COL in ex:
                ref_raw = ex[TRANSCRIPT_COL]
            elif "transcription" in ex:
                ref_raw = ex["transcription"]
            else:
                ref_raw = ""
            ref_text = kamba_normalizer.normalize(ref_raw)

            predictions.append(pred_text)
            references.append(ref_text)

    # Filter empty references
    valid = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not valid:
        logging.warning("No valid references found!")
        return {"wer": 1.0, "cer": 1.0, "predictions": predictions,
                "references": references}

    valid_preds, valid_refs = zip(*valid)

    wer = wer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))
    cer = cer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))

    # Print sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (first 10)")
    print("="*60)
    for i, (pred, ref) in enumerate(zip(valid_preds[:10], valid_refs[:10])):
        print(f"[{i}] REF : {ref}")
        print(f"[{i}] HYP : {pred}")
        print()

    return {
        "wer": wer,
        "cer": cer,
        "score": (1 - (0.5 * wer + 0.5 * cer)) * 100,
        "n_samples": len(valid_preds),
        "predictions": list(predictions),
        "references": list(references),
    }


# ── Saving results ────────────────────────────────────────────────────────────
def save_results(result: Dict, experiment_name: str, split: str):
    """Save predictions and metrics to json_outputs/."""
    out_dir = Path("json_outputs") / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions file
    pred_path = out_dir / f"predictions_{split}.json"
    output = {}
    for i, (pred, ref) in enumerate(
        zip(result["predictions"], result["references"])
    ):
        output[str(i)] = {
            "reference": ref,
            "prediction": pred,
        }
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"Predictions saved: {pred_path}")

    # Metrics file
    metrics = {
        "experiment": experiment_name,
        "split": split,
        "n_samples": result["n_samples"],
        "wer": round(result["wer"], 4),
        "cer": round(result["cer"], 4),
        "score": round(result["score"], 4),
        "wer_pct": f"{result['wer']*100:.2f}%",
        "cer_pct": f"{result['cer']*100:.2f}%",
    }
    metrics_path = out_dir / f"metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved: {metrics_path}")

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Kamba ASR model")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to saved model directory (contains processor/ and model files)"
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True,
        help="Name for output files (e.g. kamba_w2v_bert_10h)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "validation"],
        help="Which split to evaluate (test or validation)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for inference (reduce if OOM)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit evaluation to N samples (None = all)"
    )
    parser.add_argument(
        "--test_sample_size", type=int, default=500,
        help="Number of clips to stream for test split (must match training)"
    )
    parser.add_argument(
        "--val_sample_size", type=int, default=500,
        help="Number of clips for validation split (after test clips)"
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logging.info(f"Experiment: {args.experiment_name}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Split     : {args.split}")

    # Load data — stream exactly the same clips used during training
    if args.split == "test":
        dataset = load_test_split(
            test_sample_size=args.test_sample_size
        )
    else:  # validation
        logging.info("Streaming validation split...")
        stream = load_dataset(
            "DDD-Kenya/Kamba-ASR-Data-Subset-484H",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        val_rows = []
        for i, ex in enumerate(stream):
            if i < args.test_sample_size:
                continue   # skip test clips
            if len(val_rows) >= args.val_sample_size:
                break
            val_rows.append(ex)
        logging.info(f"Collected {len(val_rows)} validation clips")
        dataset = Dataset.from_list(val_rows)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Load model
    model, processor, device = load_model(args.model_path)

    # Evaluate
    result = evaluate_kamba(
        dataset=dataset,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )

    # Save and print summary
    metrics = save_results(result, args.experiment_name, args.split)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    summary = pd.DataFrame([{
        "Split"     : args.split,
        "Samples"   : metrics["n_samples"],
        "WER"       : metrics["wer_pct"],
        "CER"       : metrics["cer_pct"],
        "Score (%)" : metrics["score"],
    }])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()