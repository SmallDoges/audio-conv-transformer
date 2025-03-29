import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class AudioTransformer(nn.Module):
    """
    Transformer model for audio representation learning
    """
    def __init__(
        self,
        vocab_size=512,  # Number of discrete tokens in the codebook
        d_model=512,     # Embedding dimension
        nhead=8,         # Number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=5000
    ):
        """
        Args:
            vocab_size: Size of vocabulary (number of discrete tokens)
            d_model: Hidden dimensionality of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Dimensionality of feedforward network in transformer
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(AudioTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters of the model
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask (usually a causal mask)
            memory_mask: Memory mask
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, vocab_size]
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self._generate_square_mask(src.size(1), src.device)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        # Embed tokens
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Apply transformer
        output = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            src_mask: Source mask
            
        Returns:
            Memory tensor from the encoder
        """
        if src_mask is None:
            src_mask = self._generate_square_mask(src.size(1), src.device)
        
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        memory = self.transformer.encoder(src_emb, src_mask)
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode target sequence given memory from encoder
        
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len]
            memory: Memory from encoder [batch_size, src_seq_len, d_model]
            tgt_mask: Target mask
            memory_mask: Memory mask
            
        Returns:
            Output tensor from the decoder
        """
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        output = self.output_projection(output)
        
        return output
    
    def _generate_square_mask(self, sz, device):
        """
        Generate mask for padding (0s in the sequence)
        """
        mask = (torch.ones((sz, sz), device=device) == 1)
        return mask
    
    def _generate_square_subsequent_mask(self, sz, device):
        """
        Generate causal mask for the decoder
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device), diagonal=1) == 0)
        return mask


class AudioFeatureTransformer(nn.Module):
    """
    Transformer model for processing continuous audio features directly
    (without quantization)
    """
    def __init__(
        self,
        input_dim=80,     # Input dimension (number of mel bands)
        d_model=512,      # Model dimension
        nhead=8,          # Number of attention heads
        num_layers=6,     # Number of encoder layers
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=5000
    ):
        """
        Args:
            input_dim: Input feature dimension (number of mel bands)
            d_model: Hidden dimensionality of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimensionality of feedforward network in transformer
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(AudioFeatureTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters of the model
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [batch_size, seq_length, input_dim]
            mask: Attention mask
            
        Returns:
            Output features [batch_size, seq_length, input_dim]
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Project back to input dimension
        x = self.output_projection(x)
        
        return x 