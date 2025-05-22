import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from typing import Dict, Tuple, List, Optional, Union, Any

class VQVAEConfig(PretrainedConfig):
    """Configuration class for VQVAE model"""
    model_type = "vq_vae"
    
    def __init__(
        self,
        input_dim: int = 80,  # Mel spectrogram feature dimension
        hidden_dim: int = 512,
        codebook_size: int = 8192,
        codebook_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        num_conv_layers: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQVAE
    
    Implemented using the algorithm from "Neural Discrete Representation Learning"
    (van den Oord et al., 2017)
    """
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(codebook_size, codebook_dim))
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', torch.randn(codebook_size, codebook_dim))
        
    def forward(self, inputs, training=True):
        """
        Forward pass through the vector quantizer
        
        Args:
            inputs: Tensor of shape [batch_size, sequence_length, codebook_dim]
            training: Whether in training mode
        
        Returns:
            Dictionary containing:
            - quantized: Quantized tensor with gradients via straight-through estimator
            - encodings: One-hot encodings of the quantized indices
            - quantized_indices: Indices of the closest codebook entries
            - commitment_loss: Commitment loss
            - codebook_loss: Codebook loss (0 if using EMA update)
        """
        # Flatten input
        flat_inputs = inputs.view(-1, self.codebook_dim)
        
        # Calculate distances between inputs and codebook entries
        distances = torch.sum(flat_inputs ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * torch.matmul(flat_inputs, self.codebook.t())
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.codebook_size).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.codebook)
        
        # Reshape to match input shape
        quantized = quantized.view(inputs.shape)
        
        # Calculate loss
        if training:
            if self.decay > 0.0:
                # Update codebook using EMA
                with torch.no_grad():
                    # Update ema_count
                    self.ema_count = self.decay * self.ema_count + \
                                    (1 - self.decay) * torch.sum(encodings, dim=0)
                    
                    # Update ema_weight
                    encoded_sum = torch.matmul(encodings.t(), flat_inputs)
                    self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * encoded_sum
                    
                    # Update codebook
                    n = torch.sum(self.ema_count)
                    normalized_count = ((self.ema_count + 1e-5) / (n + self.codebook_size * 1e-5) * n)
                    self.codebook = self.ema_weight / normalized_count.unsqueeze(1)
                
                # Calculate commitment loss
                commitment_loss = F.mse_loss(inputs, quantized.detach())
                
                # No codebook loss when using EMA update
                codebook_loss = 0.0
            else:
                # Calculate commitment loss
                commitment_loss = F.mse_loss(inputs, quantized.detach())
                
                # Calculate codebook loss
                codebook_loss = F.mse_loss(quantized, inputs.detach())
        else:
            commitment_loss = torch.tensor(0.0, device=inputs.device)
            codebook_loss = torch.tensor(0.0, device=inputs.device)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return {
            'quantized': quantized,
            'encodings': encodings,
            'quantized_indices': encoding_indices,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss
        }


class VQVAE(PreTrainedModel):
    """
    Vector Quantized Variational Autoencoder for audio feature compression
    """
    config_class = VQVAEConfig
    base_model_prefix = "vq_vae"
    
    def __init__(self, config: VQVAEConfig):
        super().__init__(config)
        
        # Encoder
        encoder_layers = []
        in_channels = config.input_dim
        
        # Add convolutional layers
        for i in range(config.num_conv_layers):
            encoder_layers.extend([
                nn.Conv1d(
                    in_channels,
                    config.hidden_dim,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.kernel_size // 2
                ),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_channels = config.hidden_dim
        
        # Add final layer to match codebook dimension
        encoder_layers.append(
            nn.Conv1d(
                config.hidden_dim,
                config.codebook_dim,
                kernel_size=1
            )
        )
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay
        )
        
        # Decoder
        decoder_layers = []
        
        # Initial layer from codebook dimension
        decoder_layers.append(
            nn.Conv1d(
                config.codebook_dim,
                config.hidden_dim,
                kernel_size=1
            )
        )
        
        # Add convolutional layers
        for i in range(config.num_conv_layers):
            decoder_layers.extend([
                nn.Conv1d(
                    config.hidden_dim,
                    config.hidden_dim,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.kernel_size // 2
                ),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
        
        # Add final layer to reconstruct input
        decoder_layers.append(
            nn.Conv1d(
                config.hidden_dim,
                config.input_dim,
                kernel_size=1
            )
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.init_weights()
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input features to discrete tokens
        
        Args:
            x: Input features of shape [batch_size, input_dim, sequence_length]
            
        Returns:
            Dictionary containing:
            - quantized: Quantized representation
            - encodings: One-hot encodings of the quantized indices
            - quantized_indices: Indices of the closest codebook entries
            - commitment_loss: Commitment loss
            - codebook_loss: Codebook loss
        """
        # Pass through encoder
        z = self.encoder(x)
        
        # Permute for quantizer
        z = z.permute(0, 2, 1)  # [B, T, C]
        
        # Quantize
        quantized_outputs = self.quantizer(z, training=self.training)
        
        # Permute back
        quantized_outputs['quantized'] = quantized_outputs['quantized'].permute(0, 2, 1)  # [B, C, T]
        
        return quantized_outputs
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized representation back to features
        
        Args:
            quantized: Quantized representation of shape [batch_size, codebook_dim, sequence_length]
            
        Returns:
            Reconstructed features of shape [batch_size, input_dim, sequence_length]
        """
        # Pass through decoder
        reconstructed = self.decoder(quantized)
        
        return reconstructed
    
    def forward(
        self,
        features: torch.Tensor,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VQVAE
        
        Args:
            features: Input features of shape [batch_size, input_dim, sequence_length]
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary containing:
            - reconstructed: Reconstructed features
            - quantized: Quantized representation
            - commitment_loss: Commitment loss
            - codebook_loss: Codebook loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encode and quantize
        quantized_outputs = self.encode(features)
        
        # Decode
        reconstructed = self.decode(quantized_outputs['quantized'])
        
        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, features)
        
        # Combine losses
        loss = reconstruction_loss + \
               self.config.commitment_cost * quantized_outputs['commitment_loss'] + \
               quantized_outputs['codebook_loss']
        
        if not return_dict:
            return (
                loss,
                reconstructed,
                quantized_outputs['quantized'],
                quantized_outputs['quantized_indices']
            )
        
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss,
            'commitment_loss': quantized_outputs['commitment_loss'],
            'codebook_loss': quantized_outputs['codebook_loss'],
            'reconstructed': reconstructed,
            'quantized': quantized_outputs['quantized'],
            'quantized_indices': quantized_outputs['quantized_indices'],
        }
    
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to codebook indices
        
        Args:
            x: Input features of shape [batch_size, input_dim, sequence_length]
            
        Returns:
            Tensor of codebook indices of shape [batch_size, sequence_length]
        """
        quantized_outputs = self.encode(x)
        return quantized_outputs['quantized_indices'].view(x.size(0), -1)
    
    def decode_from_indices(self, indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """
        Decode from codebook indices
        
        Args:
            indices: Codebook indices of shape [batch_size, sequence_length]
            sequence_length: Length of the sequence
            
        Returns:
            Reconstructed features of shape [batch_size, input_dim, sequence_length]
        """
        batch_size = indices.size(0)
        
        # Convert indices to one-hot encodings
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.config.codebook_size).float()
        
        # Get quantized vectors
        quantized = torch.matmul(encodings, self.quantizer.codebook)
        quantized = quantized.view(batch_size, sequence_length, self.config.codebook_dim)
        
        # Permute for decoder
        quantized = quantized.permute(0, 2, 1)  # [B, C, T]
        
        # Decode
        reconstructed = self.decode(quantized)
        
        return reconstructed 