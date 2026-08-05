"""
HubertForCTC with convolutional adapter layers (like Wav2Vec2/MMS).

This module adds adapter layers to Hubert that are missing in the original
transformers implementation.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCTC, HubertForCTC
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Union, Tuple


class HubertAdapterLayer(nn.Module):
    """Single adapter layer - same as Wav2Vec2AdapterLayer."""
    
    def __init__(self, hidden_size: int, adapter_kernel_size: int = 3, adapter_stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            2 * hidden_size,
            adapter_kernel_size,
            stride=adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states


class HubertAdapter(nn.Module):
    """Adapter module containing multiple adapter layers."""
    
    def __init__(
        self, 
        hidden_size: int, 
        adapter_kernel_size: int = 3, 
        adapter_stride: int = 2,
        num_adapter_layers: int = 3
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HubertAdapterLayer(hidden_size, adapter_kernel_size, adapter_stride)
            for _ in range(num_adapter_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        # conv expects: (batch, hidden_size, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class HubertForCTCWithAdapter(HubertPreTrainedModel):
    """
    Hubert Model with adapter layers and CTC head.
    
    Similar to Wav2Vec2ForCTC with add_adapter=True.
    """
    
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)
        
        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        self.target_lang = target_lang
        
        # adapter config with defaults matching wav2vec2
        self.adapter_kernel_size = getattr(config, "adapter_kernel_size", 3)
        self.adapter_stride = getattr(config, "adapter_stride", 2)
        self.num_adapter_layers = getattr(config, "num_adapter_layers", 3)
        
        # create adapter
        self.adapter = HubertAdapter(
            hidden_size=config.hidden_size,
            adapter_kernel_size=self.adapter_kernel_size,
            adapter_stride=self.adapter_stride,
            num_adapter_layers=self.num_adapter_layers,
        )
        
        # output size after adapter is same as hidden_size (GLU keeps dimension)
        output_hidden_size = getattr(config, "output_hidden_size", config.hidden_size)
        
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTCWithAdapter.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        
        # initialize weights
        self.post_init()

    def freeze_feature_encoder(self):
        """Freeze the feature encoder (CNN layers)."""
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """Freeze the base model, only adapter and lm_head will be trained."""
        for param in self.hubert.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        # apply adapter layers
        hidden_states = self.adapter(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
            
            # get input lengths (for CTC)
            attention_mask = (
                attention_mask if attention_mask is not None 
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            
            # account for adapter downsampling
            for _ in range(self.num_adapter_layers):
                input_lengths = (input_lengths - 1) // self.adapter_stride + 1
            
            # mask padding tokens in labels
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            
            # ctc loss
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_hubert_with_adapter(
    pretrained_model_path: str,
    processor,
    adapter_kernel_size: int = 3,
    adapter_stride: int = 2,
    num_adapter_layers: int = 3,
    **kwargs
) -> HubertForCTCWithAdapter:
    """
    Create a HubertForCTC model with adapter layers.
    
    Args:
        pretrained_model_path: path to pretrained hubert model
        processor: Wav2Vec2Processor with tokenizer
        adapter_kernel_size: kernel size for adapter conv layers (default: 3)
        adapter_stride: stride for adapter conv layers (default: 2)
        num_adapter_layers: number of adapter layers (default: 3)
        **kwargs: additional config overrides
    
    Returns:
        HubertForCTCWithAdapter model
    """
    config = AutoConfig.from_pretrained(pretrained_model_path)
    
    # set adapter config
    config.adapter_kernel_size = adapter_kernel_size
    config.adapter_stride = adapter_stride
    config.num_adapter_layers = num_adapter_layers
    config.output_hidden_size = config.hidden_size
    
    # set common fine-tuning config
    config.vocab_size = len(processor.tokenizer)
    config.pad_token_id = processor.tokenizer.pad_token_id
    config.ctc_loss_reduction = kwargs.get("ctc_loss_reduction", "mean")
    config.ctc_zero_infinity = kwargs.get("ctc_zero_infinity", False)
    
    # set dropout values
    config.attention_dropout = kwargs.get("attention_dropout", 0.0)
    config.hidden_dropout = kwargs.get("hidden_dropout", 0.0)
    config.feat_proj_dropout = kwargs.get("feat_proj_dropout", 0.0)
    config.mask_time_prob = kwargs.get("mask_time_prob", 0.0)
    config.layerdrop = kwargs.get("layerdrop", 0.0)
    config.final_dropout = kwargs.get("final_dropout", 0.0)
    
    # load base hubert weights into our custom model
    model = HubertForCTCWithAdapter(config)
    
    # load pretrained hubert encoder weights
    pretrained = HubertModel.from_pretrained(pretrained_model_path)
    model.hubert.load_state_dict(pretrained.state_dict())
    del pretrained
    
    return model

if __name__ == "__main__":
    print("Testing HubertForCTCWithAdapter...")
    
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained("ajesujoba/AfriHuBERT")
    config.vocab_size = 100  # dummy value for testing
    config.pad_token_id = 0
    config.ctc_loss_reduction = "mean"
    config.ctc_zero_infinity = False
    config.final_dropout = 0.0
    config.adapter_kernel_size = 3
    config.adapter_stride = 2
    config.num_adapter_layers = 3
    config.output_hidden_size = config.hidden_size
    
    model = HubertForCTCWithAdapter(config)
    
    # load pretrained weights
    pretrained = HubertModel.from_pretrained("ajesujoba/AfriHuBERT")
    model.hubert.load_state_dict(pretrained.state_dict())
    
    print("\nAdapter layers:")
    for name, param in model.named_parameters():
        if "adapter" in name or "lm_head" in name:
            print(f"  {name}: {param.shape}")
    
    print("\nSuccess!")