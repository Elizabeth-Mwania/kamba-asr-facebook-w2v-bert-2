# src/utils/cache.py
# dataset caching utilities

import json
import hashlib
import logging
from pathlib import Path
from datasets import load_from_disk



def get_encode_hash(config) -> str:
    """generate hash from config params that affect encoding."""
    key_params = {
        "dataset_path": config.dataset_path,
        "pretrained_model": config.pretrained_model,
        "character_set": config.character_set,
        "add_language_tokens": getattr(config, "add_language_tokens", False),
    }
    hash_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def load_encoded_datasets(config, cache_dir: Path):
    """load cached encoded datasets if they exist."""
    
    cache_dir = Path(cache_dir)
    encode_hash = get_encode_hash(config)
    train_cache = cache_dir / f"encoded_train_{encode_hash}"
    eval_cache = cache_dir / f"encoded_eval_{encode_hash}"
    
    if train_cache.exists() and eval_cache.exists():
        logging.info(f"Loading cached encoded datasets (hash: {encode_hash})...")
        train = load_from_disk(str(train_cache))
        eval_ = load_from_disk(str(eval_cache))
        logging.info(f"Loaded {len(train)} train, {len(eval_)} eval from cache")
        return train, eval_
    
    return None, None


def save_encoded_datasets(train_dataset, eval_dataset, config, cache_dir: Path):
    """save encoded datasets to cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    encode_hash = get_encode_hash(config)
    train_cache = cache_dir / f"encoded_train_{encode_hash}"
    eval_cache = cache_dir / f"encoded_eval_{encode_hash}"
    
    logging.info(f"Saving encoded datasets to cache (hash: {encode_hash})...")
    train_dataset.save_to_disk(str(train_cache))
    eval_dataset.save_to_disk(str(eval_cache))