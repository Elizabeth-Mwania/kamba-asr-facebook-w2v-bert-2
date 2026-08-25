#!/usr/bin/env python3
"""Create one reusable, local Kamba ASR benchmark subset.

Only the selected clips are downloaded. The resulting DatasetDict contains
self-contained PCM WAV audio plus metadata -- never model-specific features --
and can therefore be shared by Whisper, CTC, and future experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List

from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset


DATASET_ID = "Digital-Divide-Data/Kamba-ASR-Data-Subset-484H"
SAMPLE_RATE = 16_000
MANIFEST_NAME = "benchmark_manifest.json"

# The Hub dataset contains Arrow shards where ``dialect`` is null in early
# shards and string in later ones.  Declaring the full schema prevents Datasets
# from inferring ``null`` from the first shard and failing part-way through a
# streaming pass.
SOURCE_FEATURES = Features({
    "Unnamed: 0": Value("int64"),
    "speaker_name": Value("string"),
    "filename": Value("string"),
    "domain": Value("string"),
    "sub_domain": Value("string"),
    "theme": Value("string"),
    "subtheme": Value("string"),
    "sentence_id": Value("string"),
    "transcripts": Value("string"),
    "dialect": Value("string"),
    "gender": Value("string"),
    "age": Value("int64"),
    "date_of_recording": Value("string"),
    "duration": Value("string"),
    "audio": Audio(),
})


def source_stream():
    """Open the source with a stable schema across all Arrow shards."""
    return load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        features=SOURCE_FEATURES,
    )


def duration_seconds(example: Dict[str, Any]) -> float:
    """Measure duration from decoded audio, one clip at a time.

    The source ``duration`` field is a recording-time string (for example
    ``21:00``), not a clip duration, so it must never be used for selection.
    """
    audio = example["audio"]
    return len(audio["array"]) / audio["sampling_rate"]


def select_rows(train_hours: float, validation_size: int, test_size: int) -> Dict[str, List[int]]:
    """Select the legacy sequential split without retaining audio in memory."""
    stream = source_stream()
    max_seconds = train_hours * 3600
    selected = {"test": [], "validation": [], "train": []}
    train_seconds = 0.0

    for index, example in enumerate(stream):
        if len(selected["test"]) < test_size:
            selected["test"].append(index)
        elif len(selected["validation"]) < validation_size:
            selected["validation"].append(index)
        elif train_seconds < max_seconds:
            seconds = duration_seconds(example)
            if seconds > 0:
                selected["train"].append(index)
                train_seconds += seconds
        else:
            break

        if index and index % 500 == 0:
            logging.info("Scanned %s rows; selected %.2f training hours", index, train_seconds / 3600)

    if len(selected["test"]) != test_size or len(selected["validation"]) != validation_size:
        raise RuntimeError("The source stream ended before the requested evaluation sets were selected.")
    if not selected["train"]:
        raise RuntimeError("No training clips were selected.")
    logging.info("Selected %d/%d/%d clips (train/validation/test).", *[len(selected[x]) for x in ("train", "validation", "test")])
    return selected


def row_generator(indices: List[int]) -> Iterator[Dict[str, Any]]:
    """Re-stream only up to the last selected row and persist each clip safely.

    Dataset.from_generator writes Arrow shards incrementally. The source's
    audio paths are Arrow-relative, so each decoded clip is encoded immediately
    to self-contained WAV bytes instead of retaining a broken external path.
    """
    wanted = set(indices)
    if not wanted:
        return
    last = max(wanted)
    stream = source_stream()
    for index, example in enumerate(stream):
        if index > last:
            break
        if index not in wanted:
            continue
        audio = example["audio"]
        # Streaming exposes an Arrow-relative filename, not a usable local
        # path.  Returning the decoded audio lets the Audio feature encode it
        # as self-contained WAV bytes in the output dataset.
        if not isinstance(audio, dict):
            raise RuntimeError(f"Row {index} could not be read as audio.")
        if "array" in audio and "sampling_rate" in audio:
            stored_audio = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        elif audio.get("bytes") is not None:
            # Recent Datasets releases can expose the original encoded bytes
            # inside Dataset.from_generator instead of decoding them first.
            stored_audio = {"bytes": audio["bytes"], "path": audio.get("path")}
        else:
            raise RuntimeError(
                f"Row {index} contains neither decoded samples nor encoded audio bytes."
            )
        yield {
            "source_index": index,
            "sentence_id": str(example.get("sentence_id", "")),
            "speaker_name": str(example.get("speaker_name", "")),
            "transcripts": str(example["transcripts"]),
            "duration_seconds": duration_seconds(example),
            "audio": stored_audio,
        }


def build_split(indices: List[int], split_name: str, cache_dir: Path) -> Dataset:
    logging.info("Writing %s split (%d clips) ...", split_name, len(indices))
    features = Features({
        "source_index": Value("int64"),
        "sentence_id": Value("string"),
        "speaker_name": Value("string"),
        "transcripts": Value("string"),
        "duration_seconds": Value("float32"),
        "audio": Audio(sampling_rate=SAMPLE_RATE),
    })
    return Dataset.from_generator(
        row_generator,
        gen_kwargs={"indices": indices},
        features=features,
        cache_dir=str(cache_dir),
        writer_batch_size=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/kamba_10h_v1")
    parser.add_argument("--train_hours", type=float, default=10.0)
    parser.add_argument("--validation_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists. Reuse it so all models use identical data.")
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = select_rows(args.train_hours, args.validation_size, args.test_size)
    build_cache = output_dir / "_build_cache"
    try:
        dataset = DatasetDict({
            name: build_split(indices, name, build_cache) for name, indices in selections.items()
        })
        dataset.save_to_disk(str(output_dir / "dataset"))
    finally:
        # This is an internal, reproducible build cache; the saved dataset is
        # the only copy retained after a successful materialization.
        if build_cache.exists():
            shutil.rmtree(build_cache)

    manifest = {
        "dataset_id": DATASET_ID,
        "selection_method": "sequential: first test, then validation, then up-to-duration train",
        "seed": None,
        "requested_train_hours": args.train_hours,
        "splits": {name: {"count": len(indices), "source_indices": indices} for name, indices in selections.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Benchmark saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
