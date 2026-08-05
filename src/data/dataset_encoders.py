# src/data/dataset_encoders.py
from typing import Dict, Tuple, List, Any, Optional
from datasets import Dataset

from transformers import (
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor
)


from abc import ABC, abstractmethod

# type alias for processor
ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor

class DatasetEncoder(ABC):
    """Base class for dataset encoding."""
    
    def __init__(self, processor: ASRProcessor):
        self.processor = processor
        self._is_w2vbert = isinstance(processor, Wav2Vec2BertProcessor)
        self._feature_key = "input_features" if self._is_w2vbert else "input_values"
    
    def _extract_features(self, features):
        """Extract features based on processor type."""
        if self._is_w2vbert:
            return features.input_features
        return features.input_values
    
    @abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a single sample."""
        pass
    
    @abstractmethod
    def batch_encode(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Encode a batch of samples."""
        pass

    def encode_dataset(
        self,
        dataset: Dataset,
        batched: bool = True,
        batch_size: int = 32,
        remove_columns: bool = True,
    ) -> Dataset:
        """Encode a dataset and prepare it for training."""

        # self.batch_encode is implemented in the child classes
        return dataset.map(
            self.batch_encode,
            batched=batched,
            batch_size=batch_size,
            num_proc=1,
            remove_columns=dataset.column_names if remove_columns else None,
            desc="Encoding dataset using ASRDatasetEncoder"
        )


class ASRDatasetEncoder(DatasetEncoder):
    """Encoder for ASR-only dataset."""
    
    def __init__(self, processor: ASRProcessor, text_column: str = "clean_transcription"):
        super().__init__(processor)
        self.text_column = text_column
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a single sample."""
        audio = sample["audio"]
        features = self.processor(audio["array"], sampling_rate=audio["sampling_rate"])
        
        sample[self._feature_key] = self._extract_features(features)[0]
        sample["length"] = len(sample[self._feature_key])
        sample["labels"] = self.processor(text=sample[self.text_column]).input_ids
        
        return sample
    
    def batch_encode(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        audio_arrays = [audio["array"] for audio in batch["audio"]]

        sampling_rate = batch["audio"][0]["sampling_rate"]
        
        features = self.processor(audio_arrays, sampling_rate=sampling_rate)
        
        batch[self._feature_key] = self._extract_features(features)
        batch["length"] = [len(f) for f in batch[self._feature_key]]

   

        batch["labels"] = self.processor(text=batch[self.text_column]).input_ids
        
        return batch


class ASRAndLIDDatasetEncoder(ASRDatasetEncoder):
    """Encoder for ASR + LID multi-task dataset."""
    
    def __init__(
        self,
        processor: ASRProcessor,
        lang2id: Dict[str, int],
        text_column: str = "clean_transcription",
        language_column: str = "language",
    ):
        super().__init__(processor, text_column)
        self.lang2id = lang2id
        self.language_column = language_column
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a single sample."""
        batch = super().__call__(batch)
        batch["lid_labels"] = self.lang2id[batch[self.language_column]]
        return batch
    
    def batch_encode(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Encode a batch of samples."""
        batch = super().batch_encode(batch)
        batch["lid_labels"] = [self.lang2id[lang] for lang in batch[self.language_column]]
        return batch