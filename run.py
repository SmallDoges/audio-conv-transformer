#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import logging
from pathlib import Path
from transformers import TrainingArguments, HfArgumentParser

from audio_transformer.data.data_loader import AudioDataModule
from audio_transformer.models.vq_vae import VQVAE, VQVAEConfig
from audio_transformer.models.transformer import AudioTransformer, AudioTransformerConfig, AudioFeatureTransformer, AudioFeatureTransformerConfig
from audio_transformer.utils.trainer import train_vqvae, train_audio_transformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Transformer")
    
    # Command options
    parser.add_argument(
        "command",
        type=str,
        choices=["train", "inference", "demo"],
        help="Command to run",
    )
    
    # Dataset options
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="Path to a single audio file for inference or demo",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="custom",
        choices=["gtzan", "fsdd", "urbansound", "custom"],
        help="Type of dataset to use",
    )
    
    # Model options
    parser.add_argument(
        "--model_type",
        type=str,
        default="vqvae",
        choices=["vqvae", "transformer", "feature_transformer"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--vqvae_checkpoint",
        type=str,
        default=None,
        help="Path to VQVAE checkpoint for transformer training or inference",
    )
    parser.add_argument(
        "--transformer_checkpoint",
        type=str,
        default=None,
        help="Path to transformer checkpoint for inference",
    )
    
    # Training options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    
    # Audio processing options
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
    
    # VQVAE options
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for models",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=8192,
        help="Size of VQ-VAE codebook",
    )
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=64,
        help="Dimensionality of VQ-VAE codebook vectors",
    )
    
    # Transformer options
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    
    # Inference options
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate audio during inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum sequence length for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    
    args = parser.parse_args()
    return args


def load_vqvae(checkpoint_path):
    """Load a pre-trained VQVAE model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VQVAE checkpoint not found at {checkpoint_path}")
    
    # Load from Hugging Face format
    logger.info(f"Loading VQVAE from {checkpoint_path}")
    model = VQVAE.from_pretrained(checkpoint_path)
    model.eval()
    return model


def load_transformer(checkpoint_path, model_type="transformer"):
    """Load a pre-trained transformer model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Transformer checkpoint not found at {checkpoint_path}")
    
    # Load from Hugging Face format
    logger.info(f"Loading {model_type} from {checkpoint_path}")
    if model_type == "transformer":
        model = AudioTransformer.from_pretrained(checkpoint_path)
    else:
        model = AudioFeatureTransformer.from_pretrained(checkpoint_path)
    model.eval()
    return model


def train(args):
    """Train a model"""
    # Create data module
    logger.info("Setting up data module")
    data_module = AudioDataModule(
        dataset_type=args.dataset_type,
        audio_dir=args.audio_dir,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
    )
    data_module.setup()
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=5,
        logging_steps=100,
        eval_steps=500,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    if args.model_type == "vqvae":
        # Train VQVAE
        logger.info("Training VQVAE model")
        
        # Create model config
        model_config = VQVAEConfig(
            input_dim=args.n_mels,
            hidden_dim=args.hidden_dim,
            codebook_size=args.codebook_size,
            codebook_dim=args.codebook_dim,
        )
        
        # Train model
        model = train_vqvae(
            data_module=data_module,
            model_config=model_config,
            training_args=training_args,
            output_dir=os.path.join(args.output_dir, "vqvae"),
        )
        
        logger.info(f"VQVAE training complete. Model saved to {os.path.join(args.output_dir, 'vqvae')}")
        
    elif args.model_type == "transformer":
        # Train Transformer
        if args.vqvae_checkpoint is None:
            raise ValueError("Must provide VQVAE checkpoint for transformer training")
        
        # Load VQVAE
        vqvae_model = load_vqvae(args.vqvae_checkpoint)
        
        # Create model config
        model_config = AudioTransformerConfig(
            vocab_size=vqvae_model.config.codebook_size,
            hidden_size=args.hidden_dim,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
        )
        
        # Train model
        logger.info("Training Transformer model")
        model = train_audio_transformer(
            data_module=data_module,
            model_config=model_config,
            vqvae_model=vqvae_model,
            training_args=training_args,
            output_dir=os.path.join(args.output_dir, "transformer"),
            use_vq=True,
        )
        
        logger.info(f"Transformer training complete. Model saved to {os.path.join(args.output_dir, 'transformer')}")
        
    elif args.model_type == "feature_transformer":
        # Train Feature Transformer (without VQ-VAE)
        
        # Create model config
        model_config = AudioFeatureTransformerConfig(
            feature_dim=args.n_mels,
            hidden_size=args.hidden_dim,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
        )
        
        # Train model
        logger.info("Training Feature Transformer model")
        model = train_audio_transformer(
            data_module=data_module,
            model_config=model_config,
            training_args=training_args,
            output_dir=os.path.join(args.output_dir, "feature_transformer"),
            use_vq=False,
        )
        
        logger.info(f"Feature Transformer training complete. Model saved to {os.path.join(args.output_dir, 'feature_transformer')}")


def inference(args):
    """Run inference"""
    # Load audio file
    if args.audio_file is None:
        raise ValueError("Must provide audio file for inference")
    
    # Setup data processing
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load audio
    logger.info(f"Loading audio from {args.audio_file}")
    audio, sr = librosa.load(args.audio_file, sr=args.sample_rate)
    
    # Process audio
    logger.info("Processing audio")
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / max(log_mel_spec.std(), 1e-8)
    
    # Convert to tensor
    mel_tensor = torch.tensor(log_mel_spec).unsqueeze(0).float()
    
    # Determine inference mode
    if args.vqvae_checkpoint:
        # Load VQVAE
        vqvae_model = load_vqvae(args.vqvae_checkpoint)
        
        # Run VQVAE inference
        with torch.no_grad():
            vqvae_outputs = vqvae_model(features=mel_tensor)
            reconstructed = vqvae_outputs["reconstructed"]
            quantized_indices = vqvae_outputs["quantized_indices"]
        
        # Visualize mel spectrograms
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.title("Original Mel Spectrogram")
        plt.imshow(log_mel_spec, aspect='auto', origin='lower')
        
        plt.subplot(3, 1, 2)
        plt.title("Reconstructed Mel Spectrogram")
        plt.imshow(reconstructed.squeeze().numpy(), aspect='auto', origin='lower')
        
        plt.subplot(3, 1, 3)
        plt.title("Quantized Indices")
        plt.imshow(quantized_indices.view(1, -1).numpy(), aspect='auto', cmap='viridis')
        
        os.makedirs("outputs", exist_ok=True)
        plt.tight_layout()
        plt.savefig("outputs/mel_reconstruction.png")
        logger.info("Saved visualization to outputs/mel_reconstruction.png")
        
        # Generate with transformer if requested
        if args.generate and args.transformer_checkpoint:
            # Load transformer
            transformer_model = load_transformer(args.transformer_checkpoint)
            
            # Generate
            logger.info("Generating with transformer")
            with torch.no_grad():
                # Get the first 100 tokens as a prompt
                prompt_length = min(100, quantized_indices.shape[1])
                prompt = quantized_indices[:, :prompt_length]
                
                # Generate
                generated = transformer_model.generate(
                    input_ids=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
                
                # Decode with VQVAE
                generated_mel = vqvae_model.decode_from_indices(
                    generated, sequence_length=generated.shape[1]
                )
            
            # Visualize
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.title("Original Mel Spectrogram")
            plt.imshow(log_mel_spec, aspect='auto', origin='lower')
            
            plt.subplot(2, 1, 2)
            plt.title("Generated Mel Spectrogram")
            plt.imshow(generated_mel.squeeze().numpy(), aspect='auto', origin='lower')
            
            plt.tight_layout()
            plt.savefig("outputs/generated_mel.png")
            logger.info("Saved generation visualization to outputs/generated_mel.png")
            
            # Convert to audio (optional)
            try:
                # Denormalize
                generated_mel_db = generated_mel.squeeze().numpy()
                generated_mel_power = librosa.db_to_power(generated_mel_db)
                
                # Griffin-Lim
                audio_generated = librosa.feature.inverse.mel_to_audio(
                    generated_mel_power,
                    sr=args.sample_rate,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length
                )
                
                # Save audio
                import soundfile as sf
                sf.write("outputs/generated_audio.wav", audio_generated, args.sample_rate)
                logger.info("Saved generated audio to outputs/generated_audio.wav")
            except Exception as e:
                logger.error(f"Error converting to audio: {e}")


def demo(args):
    """Run a simple demonstration"""
    # Similar to inference but with some preset visualizations
    inference(args)


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate function
    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    elif args.command == "demo":
        demo(args)


if __name__ == "__main__":
    main() 