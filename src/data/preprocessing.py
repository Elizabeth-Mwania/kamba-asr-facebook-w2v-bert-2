# src/data/preprocessing.py
import re
import unicodedata
from typing import Dict, List, Any, Union
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor


# type alias for processor
ASRProcessor = Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]

# Pre-compute accent replacements 
_ACCENT_REPLACEMENTS = {}

   
# REPLACE:
def clean_text_batch(batch: Dict[str, Any], 
                     allowed_chars: str = "abcdefghijklmnoprstuvwyĩũ' 0123456789",
                     apply_accent_replacements: bool = True) -> Dict[str, Any]:
    """Clean text transcripts in a batch of data.    
    Args:
        batch: dictionary containing batch data with 'transcription' field
        character_set: set of characters to keep
        apply_accent_replacements: whether to apply accent replacements
    
    Returns:
        dictionary containing batch data with 'clean_transcription' field
    """
    # convert user-provided character str to set for faster cleaning
    allowed_char_set = set(allowed_chars)
    
    # apply cleaning to all transcriptions in the batch
    batch["clean_transcription"] = [
        _clean_text_with_char_set(text, allowed_char_set, apply_accent_replacements) 
        for text in batch["transcription"]
    ]
    return batch


def _clean_text_with_char_set(text: str,
                              character_set: set[str],
                              apply_accent_replacements: bool = True) -> str:
    """Internal function that uses user-provided character set for faster cleaning"""
    
    # apply NFC normalization to ensure consistent Unicode normalization
    # example: e + combining acute accent -> é (\u00e9)
    text = unicodedata.normalize("NFC", text)
    
    # replace problematic chars with standard equivalents
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("ʼ", "'")
            .replace("`", "'")
    )

    # apply accent replacements using pre-computed dictionary
    if apply_accent_replacements:
        for src, tgt in _ACCENT_REPLACEMENTS.items():
            text = text.replace(src, tgt)

    # keep only allowed characters using user-provided character set
    text = ''.join(c for c in text if c.lower() in character_set)
    
    # normalize whitespace to single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()


def prepare_dataset(batch: Dict[str, Any], processor: ASRProcessor) -> Dict[str, Any]:
    """Prepare dataset for training by processing audio and tokenizing text.
    
    Args:
        batch: dictionary containing batch data
        processor: Wav2Vec2Processor/Wav2Vec2BertProcessor for audio/text processing
    
    Returns:
        dictionary containing batch data with 'input_features'/'input_values'/'length'/'labels' fields
    """

    # Process audio
    audio = batch["audio"]
    features = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    
    # handle both processor types in one line
    is_w2vBERT = isinstance(processor, Wav2Vec2BertProcessor)

    if is_w2vBERT: 
        key = "input_features"
        features = features.input_features[0]
    else:
        # for Wav2Vec2Processor or similar processors
        key = "input_values"
        features = features.input_values[0]
    
    batch[key] = features
    batch["length"] = len(features)
    batch["labels"] = processor(text=batch["clean_transcription"]).input_ids
    
    return batch

# batch implementation
def prepare_dataset_batch(batch: Dict[str, List[Any]], processor: ASRProcessor) -> Dict[str, List[Any]]:
    """Prepare dataset batch for training by processing audio and tokenizing text.
    
    Args:
        batch: dictionary containing batch data
        processor: Wav2Vec2Processor/Wav2Vec2BertProcessor for audio/text processing
    
    Returns:
        dictionary containing batch data with the fields:
            - 'input_features'/'input_values'
            - 'length'
            - 'labels'
    """
    # process multiple samples at once
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
    
    # process all audio at once if possible
    features = processor(audio_arrays, sampling_rate=sampling_rates[0])
    
    if isinstance(processor, Wav2Vec2BertProcessor):
        key = "input_features"
        batch[key] = features.input_features
    else:
        key = "input_values"
        batch[key] = features.input_values
    
    # adding a length key is important for speeding up training 
    # because otherwise the trainer has to compute the length 
    # of the input features/values for each sample
    batch["length"] = [len(f) for f in batch[key]]
    
    # tokenize all texts at once
    labels = processor(text=batch["clean_transcription"]).input_ids
    batch["labels"] = labels

    return batch