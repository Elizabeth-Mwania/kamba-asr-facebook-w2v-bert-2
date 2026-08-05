# src/data/dataset.py
import logging
import os
import random
from typing import Dict, Tuple, List, Any, Optional
from datasets import load_dataset, Dataset, Audio, DatasetDict

from datasets import disable_caching
disable_caching()

# for vectorized filtering on large datasets via Arrow
# NOTE: this did not work for some reason; investigate later
import pyarrow.compute as pc


from tqdm import tqdm

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor
)

from src.data.preprocessing import (
    clean_text_batch, 
    prepare_dataset_batch
)

from src.utils.config import ASRConfig

# type alias for processor
ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor


def _ensure_audio_duration_column(dataset: Dataset, split_name: str) -> Dataset:
    """
    Ensure dataset has an 'audio_duration' column, 
    renaming from 'duration' if present or computing if missing.
    """
    if "audio_duration" in dataset.column_names:
        return dataset
    # if "duration" in dataset.column_names:
    #     return dataset.rename_column("duration", "audio_duration")
    if "duration" in dataset.column_names:
        def parse_duration(example):
            raw = example["duration"]
            if isinstance(raw, str) and ':' in raw:
                parts = raw.split(':')
                example["audio_duration"] = int(parts[0]) * 60 + float(parts[1])
            else:
                try:
                    example["audio_duration"] = float(raw)
                except (ValueError, TypeError):
                    example["audio_duration"] = 5.0
            return example
        dataset = dataset.map(parse_duration, desc=f"Parsing duration in {split_name}")
        dataset = dataset.remove_columns("duration")
        return dataset

    # otherwise, compute audio duration column
    audio_duration_list = []
    for audio in tqdm(
        dataset["audio"],
        total=len(dataset["audio"]),
        desc=f"Calculating audio duration in {split_name} split",
    ):
        try:
            audio_duration_list.append(len(audio["array"]) / audio["sampling_rate"])
        except Exception as e:
            logging.error(f"Error calculating audio duration for audio {audio}: {e}")
            audio_duration_list.append(0.0)
    logging.info(f"Creating audio_duration column in {split_name} split...")

    return dataset.add_column("audio_duration", audio_duration_list)


def _ensure_transcription_column(dataset: Dataset, split_name: str) -> Dataset:
    """
    Ensure dataset has a 'transcription' column, 
    renaming from 'transcript' or 'text' if present.
    """
    if "transcription" in dataset.column_names:
        return dataset
    if "transcript" in dataset.column_names:
        return dataset.rename_column("transcript", "transcription")
    if "text" in dataset.column_names:
        return dataset.rename_column("text", "transcription")
    if "transcripts" in dataset.column_names:
        return dataset.rename_column("transcripts", "transcription")
    
    # if no transcription column is found, raise an error
    raise ValueError(
        f"Transcription column was not found in {split_name} split, "
        f"which should be called 'transcript', 'transcripts', 'text', or 'transcription'. "
        f"Found columns: {dataset.column_names}."
    )


def _create_splits_if_needed(dataset: DatasetDict, config: ASRConfig) -> DatasetDict:
    """Create deterministic train/validation/test splits from a train-only dataset.

    When ``split_group_column`` is configured, all recordings from the same group
    (for example, a speaker) remain in one split to prevent evaluation leakage.
    """
    required_splits = {config.train_split, config.eval_split, config.test_split}
    if required_splits.issubset(dataset.keys()):
        return dataset
    if not config.create_splits_if_missing:
        missing = sorted(required_splits.difference(dataset.keys()))
        raise ValueError(
            f"Dataset is missing splits {missing}. Set create_splits_if_missing: true "
            "to split its training data locally."
        )
    if config.train_split not in dataset:
        raise ValueError(f"Source split '{config.train_split}' was not found.")
    if not 0 < config.validation_split_size < 1 or not 0 < config.test_split_size < 1:
        raise ValueError("validation_split_size and test_split_size must be between 0 and 1.")
    if config.validation_split_size + config.test_split_size >= 1:
        raise ValueError("validation_split_size + test_split_size must be less than 1.")

    source = dataset[config.train_split]
    group_column = config.split_group_column
    if group_column and group_column in source.column_names:
        groups = list(source.unique(group_column))
        if len(groups) < 3:
            raise ValueError(f"Need at least three distinct values in '{group_column}' to split by group.")
        random.Random(config.seed).shuffle(groups)
        test_count = max(1, round(len(groups) * config.test_split_size))
        validation_count = max(1, round(len(groups) * config.validation_split_size))
        if test_count + validation_count >= len(groups):
            raise ValueError("Split sizes leave no groups for training.")
        test_groups = set(groups[:test_count])
        validation_groups = set(groups[test_count:test_count + validation_count])
        train_groups = set(groups[test_count + validation_count:])
        logging.info(
            "Creating speaker/group-disjoint local splits using '%s': %d train, %d validation, %d test groups.",
            group_column, len(train_groups), len(validation_groups), len(test_groups),
        )
        return DatasetDict({
            config.train_split: source.filter(lambda row: row[group_column] in train_groups, desc="Creating train split"),
            config.eval_split: source.filter(lambda row: row[group_column] in validation_groups, desc="Creating validation split"),
            config.test_split: source.filter(lambda row: row[group_column] in test_groups, desc="Creating test split"),
        })

    logging.warning("Creating row-level local splits; no usable split_group_column was configured.")
    remainder = source.train_test_split(
        test_size=config.validation_split_size + config.test_split_size, seed=config.seed
    )
    holdout = remainder["test"].train_test_split(
        test_size=config.test_split_size / (config.validation_split_size + config.test_split_size), seed=config.seed
    )
    return DatasetDict({
        config.train_split: remainder["train"],
        config.eval_split: holdout["train"],
        config.test_split: holdout["test"],
    })


def _limit_dataset_to_hours(dataset: Dataset, max_hours: float, seed: int) -> Dataset:
    """Select shuffled samples whose decoded audio duration totals no more than max_hours."""
    if max_hours <= 0:
        raise ValueError("max_train_hours must be greater than zero when set.")
    max_seconds = max_hours * 3600
    selected_indices: List[int] = []
    selected_durations: List[float] = []
    total_seconds = 0.0
    shuffled = dataset.shuffle(seed=seed)
    logging.info("Selecting at most %.2f hours of training audio using decoded durations...", max_hours)
    for index in tqdm(range(len(shuffled)), desc="Selecting training audio"):
        audio = shuffled[index]["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration <= 0 or total_seconds + duration > max_seconds:
            continue
        selected_indices.append(index)
        selected_durations.append(duration)
        total_seconds += duration
        if total_seconds >= max_seconds - 1e-9:
            break
    if not selected_indices:
        raise ValueError("No audio could be selected within max_train_hours.")
    selected = shuffled.select(selected_indices)
    if "audio_duration" in selected.column_names:
        selected = selected.remove_columns("audio_duration")
    selected = selected.add_column("audio_duration", selected_durations)
    logging.info("Selected %d samples totaling %.3f hours.", len(selected), total_seconds / 3600)
    return selected


def load_datasets(config: ASRConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    # Load custom dataset if specified
    if hasattr(config, 'use_custom_dataset') and config.use_custom_dataset:
        if hasattr(config, 'dataset_path') and config.dataset_path:
            logging.info(f"Loading custom training dataset locally from "
                         f"{config.dataset_path}...")
            
            dataset = DatasetDict.load_from_disk(config.dataset_path)
        else:
            raise ValueError(f"dataset_path to a local dataset must be specified "
                             f"when use_custom_dataset is True")

    else:
        logging.info(
            f"Loading dataset from HF hub (streaming): {config.dataset_path}"
        )
        val_count  = getattr(config, 'validation_sample_size', None) or 500
        test_count = getattr(config, 'test_sample_size', None) or 500
        max_sec    = (config.max_train_hours or 10.0) * 3600

        stream = load_dataset(
            config.dataset_path,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        train_rows, val_rows, test_rows = [], [], []
        train_sec = 0.0

        logging.info(
            f"Streaming: collecting {test_count} test + {val_count} val "
            f"+ up to {max_sec/3600:.1f}h train clips..."
        )
        def strip_audio_array(example):
            """Remove decoded float32 array but keep bytes for lazy re-decode.
            The Audio feature can reconstruct from bytes without needing a local file."""
            ex = dict(example)
            if 'audio' in ex and isinstance(ex['audio'], dict):
                audio = ex['audio']
                # Keep bytes (raw audio data) and sampling_rate, drop array only
                # bytes allows lazy decode without local file access
                slim_audio = {}
                if 'bytes' in audio and audio['bytes'] is not None:
                    slim_audio['bytes'] = audio['bytes']
                    slim_audio['path'] = audio.get('path', None)
                elif 'path' in audio:
                    # No bytes — must keep array since path is HF remote path
                    # In this case don't strip (rare edge case)
                    slim_audio = audio
                ex['audio'] = slim_audio
            return ex

        for i, example in enumerate(stream):
            if i % 500 == 0:
                logging.info(
                    f"  Scanned {i:,} | train={train_sec/3600:.2f}h "
                    f"val={len(val_rows)} test={len(test_rows)}"
                )
            if len(test_rows) < test_count:
                test_rows.append(strip_audio_array(example))
            elif len(val_rows) < val_count:
                val_rows.append(strip_audio_array(example))
            elif train_sec < max_sec:
                # Measure duration BEFORE stripping array
                try:
                    dur = len(example['audio']['array']) / (
                        example['audio']['sampling_rate'] or 16000)
                except Exception:
                    dur = 5.0
                if dur <= 0:
                    dur = 5.0
                train_rows.append(strip_audio_array(example))
                train_sec += dur
            else:
                break

        logging.info(
            f"Collected: train={len(train_rows)} clips ({train_sec/3600:.2f}h) | "
            f"val={len(val_rows)} | test={len(test_rows)}"
        )
        config._streaming_splits_done = True
        dataset = DatasetDict({
            config.train_split: Dataset.from_list(train_rows),
            config.eval_split:  Dataset.from_list(val_rows),
            config.test_split:  Dataset.from_list(test_rows),
        })
    # dataset = _create_splits_if_needed(dataset, config)
    # splits already created during streaming above — skip
    if not getattr(config, '_streaming_splits_done', False):
        dataset = _create_splits_if_needed(dataset, config)

    # cast audio column to Audio with 16000 Hz sampling rate
    logging.info(f"Casting audio column to Audio with 16000 Hz sampling rate...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    logging.info("Preparing train, validation, and test splits...")
    # if language is not "all", filter dataset to only include language
    if config.language != "all":
        languages = dataset[config.train_split].unique("language")
        if config.language not in languages:
            raise ValueError(f"Language {config.language} not found in dataset")
        
        logging.info(f"Filtering dataset for {config.language.upper()} language...")

        # table = dataset[config.train_split].data.table
        # mask = pc.equal(table.column("language"), config.language)
        # filtered_table = table.filter(mask)
        # train_dataset = Dataset(filtered_table)

        train_dataset = dataset[config.train_split].filter(
            lambda x: x["language"] == config.language,
            batch_size=32,
            desc="Filtering train split"
        )

        # table = dataset[config.eval_split].data.table
        # mask = pc.equal(table.column("language"), config.language)
        # filtered_table = table.filter(mask)
        # dev_dataset = Dataset(filtered_table)

        dev_dataset = dataset[config.eval_split].filter(
            lambda x: x["language"] == config.language,
            batch_size=32,
            desc="Filtering validation split"
        )
        test_dataset = dataset[config.test_split].filter(
            lambda x: x["language"] == config.language,
            batch_size=32,
            desc="Filtering test split"
        )
    else: # all languages => multilingual model
        train_dataset = dataset[config.train_split]
        dev_dataset = dataset[config.eval_split]
        test_dataset = dataset[config.test_split]

    train_dataset = _ensure_transcription_column(train_dataset, "train")
    dev_dataset = _ensure_transcription_column(dev_dataset, "validation")
    test_dataset = _ensure_transcription_column(test_dataset, "test")

    # if config.max_train_hours is not None:
    #     train_dataset = _limit_dataset_to_hours(train_dataset, config.max_train_hours, config.seed)
    if config.max_train_hours is not None and not getattr(config, '_streaming_splits_done', False):
        train_dataset = _limit_dataset_to_hours(train_dataset, config.max_train_hours, config.seed)

    # if there is a column called "duration" rename it to "audio_duration"
    train_dataset = _ensure_audio_duration_column(train_dataset, "train")
    dev_dataset = _ensure_audio_duration_column(dev_dataset, "validation")
    test_dataset = _ensure_audio_duration_column(test_dataset, "test")

    # sample dataset if specified
    if config.sample:
        logging.info(f"Sampling dataset to {config.sample_size} samples...")

        # shuffle the train dataset
        train_dataset = train_dataset.shuffle(seed=config.seed)
        train_dataset = train_dataset.select(range(min(config.sample_size, len(train_dataset))))

        # shuffle the dev dataset
        dev_dataset = dev_dataset.shuffle(seed=config.seed)
        # => sample 2000 sampels because valdiation set is large
        dev_dataset = dev_dataset.select(range(min(2000, len(dev_dataset))))

    # if there is a column called "audio_filepath" rename it to "audio"
    if "audio_filepath" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in train_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in train dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {train_dataset.column_names}.")
    
    # same for dev dataset  
    if "audio_filepath" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in dev_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in validation dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {dev_dataset.column_names}.")
    if "audio_filepath" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column("audio_filepath", "audio")
    elif "audio" not in test_dataset.column_names:
        raise ValueError(f"Audio filepath column was not found in test dataset. Found columns: {test_dataset.column_names}.")
    
    # Remove features not used in training 
    logging.info(f"Removing unnecessary columns...")
    features_to_keep = [
        "audio", "transcription", "audio_duration", "language",
    ]

    features_to_remove = [f for f in train_dataset.features if f not in features_to_keep]

    train_dataset = train_dataset.remove_columns(features_to_remove)
    dev_dataset = dev_dataset.remove_columns(features_to_remove)
    test_features_to_remove = [f for f in test_dataset.features if f not in features_to_keep]
    test_dataset = test_dataset.remove_columns(test_features_to_remove)
    
    # # remove samples that are longer than a max duration threshold
    # max_duration = 42.0 
    # logging.info(f"Removing samples that are longer than {max_duration} seconds...")
    # train_dataset = train_dataset.filter(
    #     lambda x: x["audio_duration"] < max_duration,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing long samples in train split"
    # )

    # dev_dataset = dev_dataset.filter(
    #     lambda x: x["audio_duration"] < max_duration,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing long samples in validation split"
    # )

    # # remove samples that are shorter than one second
    # logging.info(f"Removing samples that are shorter than one second...")
    # train_dataset = train_dataset.filter(
    #     lambda x: x["audio_duration"] > 1.0,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing short samples in train split"
    # )
    # dev_dataset = dev_dataset.filter(
    #     lambda x: x["audio_duration"] > 1.0,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing short samples in validation split"
    # )

    # Preprocess text transcripts by removing special characters
    logging.info(f"Preprocessing text transcripts...")
    train_dataset = train_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        num_proc=1,
        desc="Cleaning text transcripts in train split"
    )
    dev_dataset = dev_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        num_proc=1,
        desc="Cleaning text transcripts in validation split"
    )
    test_dataset = test_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        num_proc=1,
        desc="Cleaning text transcripts in test split"
    )

    # add language tokens to the beginning of the transcription
    if config.add_language_tokens and config.language == "all":        
        train_dataset = train_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=16,
            num_proc=1,
            desc="Adding language tags to train transcriptions"
        )

        dev_dataset = dev_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=16,
            num_proc=1,
            desc="Adding language tags to validation transcriptions"
        )

    return train_dataset, dev_dataset, test_dataset


def add_language_tag_to_transcript(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Add language tags to the beginning of the transcription"""

    # word delimiter "|" was used instead of space after language tag
    # tokenizer strips whitespace around special tags like [AMHARIC],
    # but "|" is encoded as a regular token (mapped to space in vocab)
    batch["clean_transcription"] = [
        f"[{lang.upper()}]|{trans}"
        for lang, trans in zip(batch["language"], batch["clean_transcription"])
    ]

    return batch


def build_vocabulary(character_set: set[str],
                     add_language_tags: bool = False,
                     language_tags: List[str] = None) -> Dict[str, int]:
    """Build vocabulary from user-provided character set.
    
    Args:
        character_set: Set of characters to include in the vocabulary
        add_language_tags: Whether to add language tags to the vocabulary (only for multilingual models)
        language_tags: List of language tags to add to the vocabulary (only for multilingual models)
        
    Returns:
        Vocabulary dictionary
    """
    # create vocabulary dictionary from the user-provided character set
    vocab_dict = {v: k for k, v in enumerate(sorted(character_set))}

    # handle special tokens
    # add word delimiter token and remove space token
    vocab_dict["|"] = vocab_dict[" "]

    del vocab_dict[" "]

    # add unknown token and padding token
    # find next available index
    next_idx = max(vocab_dict.values()) + 1

    vocab_dict["[UNK]"] = next_idx
    vocab_dict["[PAD]"] = next_idx + 1
    
    if add_language_tags:
        for i, tag in enumerate(language_tags):
            tag_key = f"[{tag.upper()}]"
            vocab_dict[tag_key] = next_idx + 2 + i
    
    return vocab_dict


def create_processor(
        config: ASRConfig, 
        ctc_dir: str) -> ASRProcessor:
    """Create a processor from tokenizer and feature extractor.
    
    Args:
        config: ASR configuration object
        ctc_dir: Path to directory containing CTC tokenizer
        
    Returns:
        Wav2Vec2Processor for processing audio and text
    """
    logging.info("=" * 60)
    logging.info(f"CTC tokenizer directory: {ctc_dir}")
    logging.info(f"Exists: {os.path.exists(ctc_dir)}")
    logging.info(f"Contents: {os.listdir(ctc_dir)}")
    logging.info("=" * 60)
    # initialize tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        ctc_dir,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # initialize feature extractor
    if config.pretrained_model == "facebook/w2v-bert-2.0":
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        # combine into processor
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )

    else:
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        # combine into processor
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    return processor


def prepare_datasets(train_dataset: Dataset, 
                     eval_dataset: Dataset, 
                     processor: Wav2Vec2Processor) -> Tuple[Dataset, Dataset]:
    """Prepare datasets for training by adding processed inputs.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Wav2Vec2Processor for processing audio and text
        
    Returns:
        Tuple of prepared (train_dataset, test_dataset)
    """
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=eval_dataset.column_names
    )   
    
    return train_dataset, eval_dataset
