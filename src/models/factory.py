# src/models/factory.py
from typing import Optional, Union
import torch
from transformers import Wav2Vec2ForCTC, AutoModelForCTC, Wav2Vec2Processor, AutoConfig
from src.utils.config import ASRConfig
from src.models.hubert_with_adapter import HubertForCTCWithAdapter, HubertModel


def create_asr_model(config: ASRConfig, 
                     processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    """
    Create and configure a Wav2Vec2 ASR model.
    
    Args:
        config: ASR configuration object
        processor: Wav2Vec2Processor with tokenizer information
        
    Returns:
        Configured Wav2Vec2ForCTC model
    """
    # Get pretrained model path
    pretrained_model_path = config.get_pretrained_model_path()
    
    # initialize model
    # if the pretrained_model is w2v-bert-2.0, add adapter
    # if pretrained_model_path == "facebook/w2v-bert-2.0":
    #     config.add_final_layer_adapter = True
    # else:
    #     add_final_layer_adapter = False

    # check if the pretrained model is a Hubert model
    if "hubert" in pretrained_model_path.lower():
        model = AutoModelForCTC.from_pretrained(
            pretrained_model_path,
            attention_dropout=0.00,
            hidden_dropout=0.00,
            feat_proj_dropout=0.00,
            mask_time_prob=0.00,
            layerdrop=0.00,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    else:
        model = AutoModelForCTC.from_pretrained(
            pretrained_model_path,
            attention_dropout=0.00,
            hidden_dropout=0.00,
            feat_proj_dropout=0.00,
            mask_time_prob=0.00,
            layerdrop=0.00,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            add_adapter=getattr(config, "add_final_layer_adapter", False),  
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )

    # apply freezing configuration
    # if model is based on wav2vec2, freeze feature encoder if specified
    if hasattr(model, 'freeze_feature_encoder') and config.freeze_feature_encoder:
        model.freeze_feature_encoder()

    return model


