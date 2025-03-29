#!/usr/bin/env python
import os
import argparse
import torch

from audio_transformer.train import main as train_main
from audio_transformer.inference import main as inference_main


def main():
    parser = argparse.ArgumentParser(description="Audio Transformer Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    train_parser.add_argument("--model_type", type=str, default="vqvae", 
                             choices=["vqvae", "transformer", "feature_transformer"],
                             help="Model type to train")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    train_parser.add_argument("--vqvae_checkpoint", type=str, default=None, 
                             help="Path to VQ-VAE checkpoint for transformer training")
    train_parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    train_parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands")
    train_parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    train_parser.add_argument("--hop_length", type=int, default=512, help="Hop length")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--vqvae_checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint")
    inference_parser.add_argument("--transformer_checkpoint", type=str, help="Path to transformer checkpoint")
    inference_parser.add_argument("--audio_file", type=str, help="Path to audio file for processing")
    inference_parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    inference_parser.add_argument("--generate", action="store_true", help="Generate new audio using transformer")
    inference_parser.add_argument("--max_length", type=int, default=1000, help="Maximum generation length")
    inference_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_args = [
            "--audio_dir", args.audio_dir,
            "--model_type", args.model_type,
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs),
            "--learning_rate", str(args.learning_rate),
            "--checkpoint_dir", args.checkpoint_dir,
            "--val_split", str(args.val_split),
            "--sample_rate", str(args.sample_rate),
            "--n_mels", str(args.n_mels),
            "--n_fft", str(args.n_fft),
            "--hop_length", str(args.hop_length)
        ]
        
        if args.vqvae_checkpoint:
            train_args.extend(["--vqvae_checkpoint", args.vqvae_checkpoint])
        
        # Redirect to train main
        import sys
        sys.argv = ["train.py"] + train_args
        train_main()
    
    elif args.command == "inference":
        inference_args = [
            "--vqvae_checkpoint", args.vqvae_checkpoint,
            "--output_dir", args.output_dir
        ]
        
        if args.transformer_checkpoint:
            inference_args.extend(["--transformer_checkpoint", args.transformer_checkpoint])
        
        if args.audio_file:
            inference_args.extend(["--audio_file", args.audio_file])
        
        if args.generate:
            inference_args.append("--generate")
        
        inference_args.extend(["--max_length", str(args.max_length)])
        inference_args.extend(["--temperature", str(args.temperature)])
        
        # Redirect to inference main
        import sys
        sys.argv = ["inference.py"] + inference_args
        inference_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 