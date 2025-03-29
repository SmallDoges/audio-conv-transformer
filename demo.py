#!/usr/bin/env python
"""
Demo script to demonstrate the audio transformer pipeline.

Usage:
    python demo.py --audio_file path/to/audio.wav

This will:
1. Load and preprocess the audio file
2. Extract Mel spectrogram features
3. Visualize the spectrograms
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import torchaudio

from audio_transformer.utils.audio_processing import AudioProcessor


def process_and_visualize(audio_file, output_dir="demo_output"):
    """
    Process an audio file and visualize its features
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Load and process audio
    waveform, sample_rate = audio_processor.load_audio(audio_file)
    
    # Display audio info
    print(f"Audio file: {audio_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
    print(f"Channels: {waveform.shape[0]}")
    
    # Convert to mono
    mono_waveform = audio_processor.convert_to_mono(waveform)
    
    # Extract Mel spectrogram
    mel_spec = audio_processor.extract_mel_spectrogram(mono_waveform)
    
    # Extract FFT frames
    fft_frames = audio_processor.extract_fft(mono_waveform)
    
    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(mono_waveform.squeeze().numpy())
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(output_dir, "waveform.png"))
    plt.close()
    
    # Plot Mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower')
    plt.title("Mel Spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel Bin")
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(os.path.join(output_dir, "mel_spectrogram.png"))
    plt.close()
    
    # Plot FFT frames
    plt.figure(figsize=(10, 4))
    plt.imshow(fft_frames.numpy(), aspect='auto', origin='lower')
    plt.title("FFT Frames")
    plt.xlabel("Frame")
    plt.ylabel("Frequency Bin")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "fft_frames.png"))
    plt.close()
    
    print(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audio Transformer Demo")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output_dir", type=str, default="demo_output", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    process_and_visualize(
        audio_file=args.audio_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main() 