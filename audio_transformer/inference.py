import os
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.audio_processing import AudioProcessor
from models.vq_vae import VQVAE, VQVAEConfig
from models.transformer import AudioTransformer, AudioFeatureTransformer, AudioTransformerConfig, AudioFeatureTransformerConfig


def plot_spectrogram(spec, title=None, save_path=None):
    """
    Plot a spectrogram
    
    Args:
        spec: Spectrogram tensor of shape [n_mels, time]
        title: Title for the plot
        save_path: Path to save the plot
    """
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec, aspect='auto', origin='lower', interpolation='none')
    
    if title:
        ax.set_title(title)
    
    ax.set_ylabel('Mel bins')
    ax.set_xlabel('Frames')
    
    fig.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def visualize_reconstructions(
    model,
    audio_file,
    audio_processor,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs"
):
    """
    Visualize the reconstructions from the VQ-VAE model
    
    Args:
        model: Trained VQ-VAE model
        audio_file: Path to audio file
        audio_processor: Audio processor instance
        device: Device to run model on
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    
    # Process audio
    waveform, _ = audio_processor.load_audio(audio_file)
    mono_waveform = audio_processor.convert_to_mono(waveform)
    mel_spec = audio_processor.extract_mel_spectrogram(mono_waveform)
    
    # Move to device
    mel_spec = mel_spec.to(device)
    
    # Reconstruct
    with torch.no_grad():
        reconstructed, _, encoding_indices = model(mel_spec)
    
    # Plot original and reconstructed spectrograms
    plot_spectrogram(
        mel_spec.squeeze().cpu(),
        title="Original Mel Spectrogram",
        save_path=os.path.join(output_dir, "original_mel.png")
    )
    
    plot_spectrogram(
        reconstructed.squeeze().cpu(),
        title="Reconstructed Mel Spectrogram",
        save_path=os.path.join(output_dir, "reconstructed_mel.png")
    )
    
    # Plot token indices
    plt.figure(figsize=(10, 4))
    plt.imshow(encoding_indices.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Token Index')
    plt.title('VQ Token Indices')
    plt.xlabel('Frame')
    plt.ylabel('Batch')
    plt.savefig(os.path.join(output_dir, "vq_tokens.png"))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    return {
        "original_mel": mel_spec.squeeze().cpu(),
        "reconstructed_mel": reconstructed.squeeze().cpu(),
        "encoding_indices": encoding_indices.cpu()
    }


def generate_from_transformer(
    transformer_model,
    vqvae_model,
    start_tokens,
    max_length=1000,
    temperature=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate a sequence from the transformer model
    
    Args:
        transformer_model: Trained transformer model
        vqvae_model: Trained VQ-VAE model
        start_tokens: Starting token sequence [batch_size, seq_len]
        max_length: Maximum generation length
        temperature: Sampling temperature
        device: Device to run model on
        
    Returns:
        generated_tokens: Generated token sequence
        generated_mel: Generated mel spectrogram
    """
    transformer_model = transformer_model.to(device)
    vqvae_model = vqvae_model.to(device)
    
    transformer_model.eval()
    vqvae_model.eval()
    
    start_tokens = start_tokens.to(device)
    batch_size = start_tokens.size(0)
    
    current_tokens = start_tokens
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get prediction
            outputs = transformer_model(
                input_ids=current_tokens,
                attention_mask=torch.ones_like(current_tokens),
                return_dict=True
            )
            
            # Get next token distribution (last position)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Decode to mel spectrogram using VQ-VAE
        sequence_length = current_tokens.size(1)
        generated_mel = vqvae_model.decode_from_indices(current_tokens, sequence_length)
    
    return current_tokens, generated_mel


def process_audio(
    vqvae_checkpoint,
    transformer_checkpoint=None,
    audio_file=None,
    output_dir="outputs",
    device="cuda" if torch.cuda.is_available() else "cpu",
    generate=False,
    max_length=1000,
    temperature=1.0
):
    """
    Process audio with the trained models
    
    Args:
        vqvae_checkpoint: Path to VQ-VAE checkpoint
        transformer_checkpoint: Path to transformer checkpoint
        audio_file: Path to audio file
        output_dir: Directory to save outputs
        device: Device to run model on
        generate: Whether to generate new audio
        max_length: Maximum generation length
        temperature: Sampling temperature
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Load VQ-VAE model
    vqvae_config = VQVAEConfig(
        input_dim=80,  # Mel bands
        hidden_dim=128,
        codebook_size=512,
        codebook_dim=64,
        num_conv_layers=3
    )
    vqvae_model = VQVAE(vqvae_config)
    
    vqvae_checkpoint = torch.load(vqvae_checkpoint, map_location=device)
    vqvae_model.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae_model.to(device)
    vqvae_model.eval()
    
    # Process and visualize audio with VQ-VAE
    if audio_file:
        results = visualize_reconstructions(
            model=vqvae_model,
            audio_file=audio_file,
            audio_processor=audio_processor,
            device=device,
            output_dir=output_dir
        )
        encoding_indices = results["encoding_indices"]
    
    # Generate new audio with transformer if requested
    if generate and transformer_checkpoint:
        # Load transformer model
        transformer_config = AudioTransformerConfig(
            vocab_size=512,  # Should match VQ-VAE codebook_size
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        transformer_model = AudioTransformer(transformer_config)
        
        transformer_checkpoint = torch.load(transformer_checkpoint, map_location=device)
        transformer_model.load_state_dict(transformer_checkpoint["model_state_dict"])
        transformer_model.to(device)
        transformer_model.eval()
        
        # Generate from transformer
        if audio_file:
            # Use encoded audio as starting point
            start_tokens = encoding_indices.unsqueeze(0)[:, :10]  # Use first 10 tokens as seed
        else:
            # Random start tokens
            start_tokens = torch.randint(1, 512, (1, 1), device=device)
        
        generated_tokens, generated_mel = generate_from_transformer(
            transformer_model=transformer_model,
            vqvae_model=vqvae_model,
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        # Plot generated spectrogram
        plot_spectrogram(
            generated_mel.squeeze().cpu(),
            title="Generated Mel Spectrogram",
            save_path=os.path.join(output_dir, "generated_mel.png")
        )
        
        # Try to convert back to audio using Griffin-Lim in torchaudio
        try:
            # Normalize spectrogram
            normalized_mel = (generated_mel.squeeze().cpu() - generated_mel.squeeze().cpu().mean()) / generated_mel.squeeze().cpu().std()
            
            # Convert to power spectrogram
            power_spec = 10 ** normalized_mel
            
            # Griffin-Lim reconstruction requires MelScale to invert
            mel_scale = torchaudio.transforms.InverseMelScale(
                n_stft=audio_processor.n_fft // 2 + 1,
                n_mels=audio_processor.n_mels,
                sample_rate=audio_processor.sample_rate
            )
            
            # Invert mel scale
            spec = mel_scale(power_spec.unsqueeze(0))
            
            # Griffin-Lim
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=audio_processor.n_fft,
                hop_length=audio_processor.hop_length,
                power=2.0
            )
            
            waveform = griffin_lim(spec)
            
            # Save audio
            output_path = os.path.join(output_dir, "generated_audio.wav")
            torchaudio.save(output_path, waveform, audio_processor.sample_rate)
            print(f"Generated audio saved to {output_path}")
        except Exception as e:
            print(f"Error generating audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Audio model inference")
    parser.add_argument("--vqvae_checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--transformer_checkpoint", type=str, help="Path to transformer checkpoint")
    parser.add_argument("--audio_file", type=str, help="Path to audio file for processing")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--generate", action="store_true", help="Generate new audio using transformer")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    process_audio(
        vqvae_checkpoint=args.vqvae_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        generate=args.generate,
        max_length=args.max_length,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main() 