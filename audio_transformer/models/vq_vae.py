import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE
    
    Implementation inspired by:
    https://github.com/pytorch/examples/blob/master/vae/main.py
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimensionality of embeddings
            commitment_cost: Commitment cost used in loss term
        """
        super(VectorQuantizer, self).__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        # Initialize embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
    
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [batch_size, embedding_dim, time]
            
        Returns:
            quantized: Tensor of shape [batch_size, embedding_dim, time]
            loss: VQ loss
            encodings: One-hot encodings with shape [batch_size, time, num_embeddings]
            encoding_indices: Tensor of indices into the codebook [batch_size, time]
        """
        # Convert inputs from BDHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        # Reshape encoding indices to [batch_size, time]
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return quantized, loss, encodings, encoding_indices


class Encoder(nn.Module):
    """
    Encoder for VQ-VAE
    """
    def __init__(self, in_channels, hidden_dims, embedding_dim):
        """
        Args:
            in_channels: Number of input channels (mel bands)
            hidden_dims: List of hidden dimensions
            embedding_dim: Dimensionality of the latent embedding
        """
        super(Encoder, self).__init__()
        
        modules = []
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        modules.append(
            nn.Conv1d(hidden_dims[-1], embedding_dim, kernel_size=1, stride=1)
        )
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, mel_bands, time]
            
        Returns:
            Encoder output of shape [batch_size, embedding_dim, time]
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder for VQ-VAE
    """
    def __init__(self, embedding_dim, hidden_dims, out_channels):
        """
        Args:
            embedding_dim: Dimensionality of the latent embedding
            hidden_dims: List of hidden dimensions (in reverse order)
            out_channels: Number of output channels (mel bands)
        """
        super(Decoder, self).__init__()
        
        modules = []
        
        modules.append(
            nn.Conv1d(embedding_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        )
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=1, padding=1)
                )
            )
        
        modules.append(
            nn.Sequential(
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Conv1d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1)
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, embedding_dim, time]
            
        Returns:
            Decoder output of shape [batch_size, out_channels, time]
        """
        return self.decoder(x)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder
    """
    def __init__(
        self,
        in_channels=80,  # Typically number of Mel bands
        hidden_dims=[128, 256],
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    ):
        """
        Args:
            in_channels: Number of input channels (mel bands)
            hidden_dims: List of hidden dimensions
            embedding_dim: Dimensionality of the latent embedding
            num_embeddings: Number of embeddings in codebook
            commitment_cost: Commitment cost for VQ loss
        """
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dims[::-1], in_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, mel_bands, time]
            
        Returns:
            reconstructed: Reconstructed input of shape [batch_size, mel_bands, time]
            vq_loss: Vector quantization loss
            encoding_indices: Indices of codebook entries [batch_size, time]
        """
        # Encode
        z = self.encoder(x)
        
        # Vector quantization
        quantized, vq_loss, _, encoding_indices = self.vq(z)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed, vq_loss, encoding_indices
    
    def encode(self, x):
        """
        Encode input to discrete tokens
        
        Args:
            x: Input tensor of shape [batch_size, mel_bands, time]
            
        Returns:
            encoding_indices: Indices of codebook entries [batch_size, time]
        """
        z = self.encoder(x)
        _, _, _, encoding_indices = self.vq(z)
        return encoding_indices
    
    def decode(self, indices):
        """
        Decode from discrete tokens
        
        Args:
            indices: Indices of codebook entries [batch_size, time]
            
        Returns:
            Decoded output of shape [batch_size, mel_bands, time]
        """
        batch_size, time = indices.shape
        
        # Convert indices to one-hot
        encodings = torch.zeros(batch_size, time, self._num_embeddings, device=indices.device)
        encodings.scatter_(2, indices.unsqueeze(-1), 1)
        
        # Convert to embeddings
        quantized = torch.matmul(encodings, self.vq._embedding.weight)
        quantized = quantized.permute(0, 2, 1)  # [batch_size, embedding_dim, time]
        
        # Decode
        return self.decoder(quantized) 