#!/usr/bin/env python3
import os
import argparse
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import soundfile as sf
import logging

from audio_transformer.models.vq_vae import VQVAE, VQVAEConfig
from audio_transformer.utils.audio_processing import load_and_process_audio as utils_load_and_process_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Transformer Demo")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file for processing",
    )
    parser.add_argument(
        "--vqvae_checkpoint",
        type=str,
        default=None,
        help="Path to VQVAE checkpoint (optional)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="FFT size",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Hop length",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel bands",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_outputs",
        help="Directory to save outputs",
    )
    
    args = parser.parse_args()
    return args


def process_audio_file(file_path, sample_rate=22050, n_fft=1024, hop_length=512, n_mels=80):
    """Load and process audio file into mel spectrogram"""
    # Load audio
    logger.info(f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    # Extract mel spectrogram
    logger.info("Extracting mel spectrogram")
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / max(log_mel_spec.std(), 1e-8)
    
    return audio, log_mel_spec


def visualize_mel(mel_spec, title="Mel Spectrogram", save_path=None):
    """Visualize mel spectrogram"""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process audio
    audio, mel_spec = process_audio_file(
        args.audio_file,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Visualize original mel spectrogram
    visualize_mel(
        mel_spec,
        title="Original Mel Spectrogram",
        save_path=os.path.join(args.output_dir, "original_mel.png")
    )
    
    # Convert mel spectrogram to tensor
    mel_tensor = torch.tensor(mel_spec).unsqueeze(0).float()
    
    # Process with VQVAE if checkpoint is provided
    if args.vqvae_checkpoint:
        logger.info(f"Loading VQVAE from {args.vqvae_checkpoint}")
        try:
            # Load VQVAE model
            vqvae = VQVAE.from_pretrained(args.vqvae_checkpoint)
            vqvae.eval()
            
            # Process with VQVAE
            with torch.no_grad():
                outputs = vqvae(features=mel_tensor)
                
                # Get reconstructed mel spectrogram
                reconstructed = outputs["reconstructed"].squeeze().numpy()
                
                # Get quantized indices
                indices = outputs["quantized_indices"].view(1, -1).numpy()
            
            # Visualize reconstructed mel spectrogram
            visualize_mel(
                reconstructed,
                title="Reconstructed Mel Spectrogram",
                save_path=os.path.join(args.output_dir, "reconstructed_mel.png")
            )
            
            # Visualize quantized indices
            plt.figure(figsize=(10, 2))
            plt.imshow(indices, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title("Quantized Indices")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "quantized_indices.png"))
            plt.close()
            
            # Convert reconstructed mel spectrogram to audio
            logger.info("Converting reconstructed mel spectrogram to audio")
            reconstructed_db = reconstructed
            reconstructed_power = librosa.db_to_power(reconstructed_db)
            
            # Griffin-Lim algorithm to convert mel spectrogram to audio
            audio_reconstructed = librosa.feature.inverse.mel_to_audio(
                reconstructed_power,
                sr=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length
            )
            
            # Save reconstructed audio
            sf.write(
                os.path.join(args.output_dir, "reconstructed_audio.wav"),
                audio_reconstructed,
                args.sample_rate
            )
            logger.info(f"Saved reconstructed audio to {os.path.join(args.output_dir, 'reconstructed_audio.wav')}")
            
        except Exception as e:
            logger.error(f"Error processing with VQVAE: {e}")
    
    # Create a comparative visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Mel Spectrogram")
    
    if args.vqvae_checkpoint:
        plt.subplot(2, 1, 2)
        plt.imshow(reconstructed, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Reconstructed Mel Spectrogram")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison.png"))
    logger.info(f"Saved comparison visualization to {os.path.join(args.output_dir, 'comparison.png')}")
    plt.close()
    
    logger.info(f"Demo complete. All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main() 