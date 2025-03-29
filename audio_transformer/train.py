import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from utils.audio_processing import AudioProcessor
from models.vq_vae import VQVAE
from models.transformer import AudioTransformer, AudioFeatureTransformer


class AudioDataset(Dataset):
    """
    Dataset for loading and processing audio files
    """
    def __init__(self, audio_dir, sample_rate=22050, max_length=None, audio_processor=None):
        """
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            max_length: Maximum audio length in samples (for padding/truncation)
            audio_processor: AudioProcessor instance for feature extraction
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Initialize audio processor if not provided
        if audio_processor is None:
            self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        else:
            self.audio_processor = audio_processor
        
        # Get all audio files
        self.audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            self.audio_files.extend(list(Path(audio_dir).glob(f'**/*{ext}')))
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Load and process audio file
        
        Returns:
            mel_spec: Mel spectrogram [n_mels, time]
        """
        audio_path = str(self.audio_files[idx])
        
        # Load and process audio
        waveform, _ = self.audio_processor.load_audio(audio_path)
        
        # Handle max length
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                # Truncate
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                # Zero-pad
                padding = torch.zeros(waveform.shape[0], self.max_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
        
        # Convert to mono and extract mel spectrogram
        mel_spec = self.audio_processor.process_audio(waveform=waveform)
        
        return mel_spec.squeeze(0)  # Remove batch dimension


def train_vqvae(
    model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    log_interval=10
):
    """
    Train the VQ-VAE model
    
    Args:
        model: VQVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save checkpoints
        log_interval: Interval for logging
    """
    print(f"Training VQ-VAE on {device}")
    model = model.to(device)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Reconstruction loss
    mse_loss = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_vq_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, mel_spec in enumerate(pbar):
            mel_spec = mel_spec.to(device)
            
            # Forward pass
            reconstructed, vq_loss, _ = model(mel_spec)
            
            # Compute reconstruction loss
            recon_loss = mse_loss(reconstructed, mel_spec)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item()
            })
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_vq_loss /= len(train_loader)
        
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Recon Loss: {train_recon_loss:.6f}, "
              f"VQ Loss: {train_vq_loss:.6f}")
        
        # Validate
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_vq_loss = 0
            
            with torch.no_grad():
                for mel_spec in val_loader:
                    mel_spec = mel_spec.to(device)
                    
                    # Forward pass
                    reconstructed, vq_loss, _ = model(mel_spec)
                    
                    # Compute reconstruction loss
                    recon_loss = mse_loss(reconstructed, mel_spec)
                    
                    # Total loss
                    loss = recon_loss + vq_loss
                    
                    # Update losses
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_vq_loss += vq_loss.item()
            
            # Average losses
            val_loss /= len(val_loader)
            val_recon_loss /= len(val_loader)
            val_vq_loss /= len(val_loader)
            
            val_losses.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.6f}, "
                  f"Recon Loss: {val_recon_loss:.6f}, "
                  f"VQ Loss: {val_vq_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_loader is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "vqvae_loss.png"))
    
    return model


def train_transformer(
    model,
    vqvae_model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    log_interval=10,
    teacher_forcing_ratio=0.5
):
    """
    Train the Transformer model
    
    Args:
        model: AudioTransformer model
        vqvae_model: Trained VQ-VAE model for encoding audio to discrete tokens
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save checkpoints
        log_interval: Interval for logging
        teacher_forcing_ratio: Ratio for teacher forcing during training
    """
    print(f"Training Transformer on {device}")
    model = model.to(device)
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()  # Set VQVAE to evaluation mode
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding token
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, mel_spec in enumerate(pbar):
            mel_spec = mel_spec.to(device)
            
            # Encode mel spectrogram to discrete tokens using VQ-VAE
            with torch.no_grad():
                tokens = vqvae_model.encode(mel_spec)
            
            # Prepare input and target for Transformer
            # For causal language modeling, input is tokens[:-1] and target is tokens[1:]
            src_tokens = tokens[:, :-1]
            tgt_tokens = tokens[:, 1:]
            
            # Create masks
            src_mask = None  # We'll use the default mask
            tgt_mask = model._generate_square_subsequent_mask(tgt_tokens.size(1), device)
            
            # Forward pass
            output = model(src_tokens, src_tokens, src_mask, tgt_mask)
            
            # Compute loss
            output = output.view(-1, model.vocab_size)
            tgt_tokens = tgt_tokens.reshape(-1)
            loss = criterion(output, tgt_tokens)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for mel_spec in val_loader:
                    mel_spec = mel_spec.to(device)
                    
                    # Encode mel spectrogram to discrete tokens using VQ-VAE
                    tokens = vqvae_model.encode(mel_spec)
                    
                    # Prepare input and target for Transformer
                    src_tokens = tokens[:, :-1]
                    tgt_tokens = tokens[:, 1:]
                    
                    # Create masks
                    src_mask = None
                    tgt_mask = model._generate_square_subsequent_mask(tgt_tokens.size(1), device)
                    
                    # Forward pass
                    output = model(src_tokens, src_tokens, src_mask, tgt_mask)
                    
                    # Compute loss
                    output = output.view(-1, model.vocab_size)
                    tgt_tokens = tgt_tokens.reshape(-1)
                    loss = criterion(output, tgt_tokens)
                    
                    # Update loss
                    val_loss += loss.item()
            
            # Average loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_loader is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "transformer_loss.png"))
    
    return model


def train_feature_transformer(
    model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    log_interval=10
):
    """
    Train the AudioFeatureTransformer model directly on audio features
    
    Args:
        model: AudioFeatureTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save checkpoints
        log_interval: Interval for logging
    """
    print(f"Training AudioFeatureTransformer on {device}")
    model = model.to(device)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, mel_spec in enumerate(pbar):
            mel_spec = mel_spec.to(device)
            
            # Forward pass
            output = model(mel_spec.permute(0, 2, 1))  # [batch, time, features]
            
            # Compute loss
            loss = criterion(output, mel_spec.permute(0, 2, 1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for mel_spec in val_loader:
                    mel_spec = mel_spec.to(device)
                    
                    # Forward pass
                    output = model(mel_spec.permute(0, 2, 1))  # [batch, time, features]
                    
                    # Compute loss
                    loss = criterion(output, mel_spec.permute(0, 2, 1))
                    
                    # Update loss
                    val_loss += loss.item()
            
            # Average loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"feature_transformer_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_loader is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "feature_transformer_loss.png"))
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train audio models")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--model_type", type=str, default="vqvae", choices=["vqvae", "transformer", "feature_transformer"],
                       help="Model type to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None, help="Path to VQ-VAE checkpoint for transformer training")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length")
    
    args = parser.parse_args()
    
    # Create audio processor
    audio_processor = AudioProcessor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Create dataset
    dataset = AudioDataset(args.audio_dir, args.sample_rate, audio_processor=audio_processor)
    
    # Split dataset into training and validation
    dataset_size = len(dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_type == "vqvae":
        # Train VQ-VAE
        model = VQVAE(
            in_channels=args.n_mels,
            hidden_dims=[128, 256],
            embedding_dim=64,
            num_embeddings=512
        )
        
        train_vqvae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.model_type == "transformer":
        # Load VQ-VAE checkpoint
        if args.vqvae_checkpoint is None:
            raise ValueError("VQ-VAE checkpoint path must be provided for transformer training")
        
        vqvae_model = VQVAE(
            in_channels=args.n_mels,
            hidden_dims=[128, 256],
            embedding_dim=64,
            num_embeddings=512
        )
        
        checkpoint = torch.load(args.vqvae_checkpoint, map_location=device)
        vqvae_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create and train transformer
        model = AudioTransformer(
            vocab_size=512,  # Should match VQ-VAE num_embeddings
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        train_transformer(
            model=model,
            vqvae_model=vqvae_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.model_type == "feature_transformer":
        # Train feature transformer directly on audio features
        model = AudioFeatureTransformer(
            input_dim=args.n_mels,
            d_model=512,
            nhead=8,
            num_layers=6
        )
        
        train_feature_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )


if __name__ == "__main__":
    main() 