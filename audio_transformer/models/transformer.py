import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from transformers import (
    PreTrainedModel, 
    BertConfig, 
    BertModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    GPT2Config, 
    GPT2Model,
    PretrainedConfig
)
from typing import Optional, Tuple, Dict, Any, Union


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_seq_length=5000):
        """
        Args:
            d_model: Hidden dimensionality of the model
            max_seq_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be part of the module's state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Output tensor with added positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class AudioTransformerConfig(PretrainedConfig):
    """Configuration class for AudioTransformer model"""
    model_type = "audio_transformer"
    
    def __init__(
        self,
        vocab_size: int = 8192,  # Size of the codebook for VQ-VAE
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_vq: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_vq = use_vq


class AudioTransformer(PreTrainedModel):
    """
    Transformer model for audio processing
    Uses BERT architecture from transformers library
    
    Works with discrete tokens from VQ-VAE
    """
    config_class = AudioTransformerConfig
    base_model_prefix = "audio_transformer"
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__(config)
        
        # Create BERT config for the underlying model
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=0,
        )
        
        # Use a pre-trained BERT model
        self.bert = BertModel(bert_config)
        
        # Output layers
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        # Pass through output layer
        logits = self.output(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for autoregressive training
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fn(
                shifted_logits.view(-1, self.config.vocab_size),
                shifted_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens autoregressively"""
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids)
        
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Clone so we don't modify the original logits
                next_token_logits_for_ban = next_token_logits.clone()
                
                # Get previous tokens
                prev_tokens = input_ids
                
                # Update logits for previously seen tokens
                for batch_idx in range(batch_size):
                    for token_idx in prev_tokens[batch_idx].unique():
                        if token_idx != 0:  # Don't penalize padding tokens
                            next_token_logits_for_ban[batch_idx, token_idx] /= repetition_penalty
                
                next_token_logits = next_token_logits_for_ban
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float("Inf")
            
            # Sample from the filtered distribution
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add generated tokens to input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens.unsqueeze(-1))], dim=-1)
            
            # Update current length
            cur_len = input_ids.shape[1]
        
        return input_ids


class AudioFeatureTransformerConfig(PretrainedConfig):
    """Configuration class for AudioFeatureTransformer model"""
    model_type = "audio_feature_transformer"
    
    def __init__(
        self,
        feature_dim: int = 80,  # Mel spectrogram feature dimension
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class AudioFeatureTransformer(PreTrainedModel):
    """
    Transformer model for audio processing that works directly with continuous features
    Uses Wav2Vec2 architecture from transformers library
    """
    config_class = AudioFeatureTransformerConfig
    base_model_prefix = "audio_feature_transformer"
    
    def __init__(self, config: AudioFeatureTransformerConfig):
        super().__init__(config)
        
        # Create Wav2Vec2 config for the underlying model
        wav2vec2_config = Wav2Vec2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            vocab_size=1,  # Not really used for feature processing
        )
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Use Wav2Vec2 model (without feature extraction layers)
        self.transformer = Wav2Vec2Model(wav2vec2_config)
        
        # Output layers for regression
        self.output = nn.Linear(config.hidden_size, config.feature_dim)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Initialize weights
        self.init_weights()
        
        # Register buffer for position ids
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length, _ = features.size()
        
        # Create position ids
        position_ids = self.position_ids[:, :seq_length]
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Project features to hidden size
        hidden_states = self.feature_projection(features)
        
        # Add position embeddings
        hidden_states = hidden_states + position_embeddings
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=features.device)
        
        # Extend the attention mask for the transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=return_dict,
        )
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        # Pass through output layer
        predicted_features = self.output(hidden_states)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(predicted_features, labels)
        
        if not return_dict:
            output = (predicted_features,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return {
            "loss": loss,
            "predicted_features": predicted_features,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def generate(
        self,
        features: torch.Tensor,
        max_length: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """Generate features autoregressively"""
        batch_size = features.shape[0]
        cur_len = features.shape[1]
        feature_dim = features.shape[2]
        
        # Initialize attention mask
        attention_mask = torch.ones((batch_size, cur_len), device=features.device)
        
        # Initialize generated features
        generated_features = features.clone()
        
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = {
                "features": generated_features,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            next_features = outputs["predicted_features"][:, -1:, :]
            
            # Add generated features
            generated_features = torch.cat([generated_features, next_features], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=features.device)],
                dim=1
            )
            
            # Update current length
            cur_len = generated_features.shape[1]
        
        return generated_features 