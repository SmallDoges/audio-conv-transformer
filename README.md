# Audio Transformer

[English](README.md) | [中文](README_CN.md)

An audio processing pipeline that transforms audio into discrete representations and processes them with a Transformer model.

## Pipeline

1. Audio Processing:
   - Convert multi-channel audio to mono
   - Frame segmentation and FFT
   - Discrete spectrum extraction
   - Mel filtering
   - Discrete Mel spectrum (frequency domain discretization)

2. Optional Vector Quantization:
   - VQ-VAE/clustering for more compact discrete tokens

3. Transformer Model:
   - Sequence length = number of frames
   - Token dimension = feature dimension

## Architecture Details

### Audio Processing
- The `AudioProcessor` class handles audio loading, resampling, and feature extraction
- Audio is converted to mono and processed into mel spectrograms
- Default parameters: 22050Hz sample rate, 1024 FFT size, 512 hop length, 80 mel bands

### Vector Quantization (VQ-VAE)
- Compresses continuous mel spectrograms into discrete tokens
- Consists of an encoder, vector quantizer, and decoder
- Encoder: Conv1D layers with BatchNorm and ReLU activations
- Vector Quantizer: Maps continuous vectors to nearest entries in a learned codebook
- Decoder: Reconstructs mel spectrograms from quantized representations

### Transformer Models
- Two implementations:
  1. `AudioTransformer`: Works with discrete tokens from VQ-VAE
  2. `AudioFeatureTransformer`: Works directly with continuous features

- The standard architecture uses:
  - 512-dimensional embeddings
  - 8 attention heads
  - 6 encoder layers
  - 6 decoder layers (for full Transformer)
  - Positional encoding for sequence awareness

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-conv-transformer.git
cd audio-conv-transformer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo
Run the demo script to test audio processing and visualization:
```bash
python demo.py --audio_file path/to/audio.wav
```

### Training
Train the VQ-VAE model:
```bash
python run.py train --audio_dir path/to/audio_files --model_type vqvae --batch_size 16 --num_epochs 100
```

Train the Transformer model (after training VQ-VAE):
```bash
python run.py train --audio_dir path/to/audio_files --model_type transformer --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt
```

Train the Feature Transformer (without VQ-VAE):
```bash
python run.py train --audio_dir path/to/audio_files --model_type feature_transformer
```

### Inference
Process an audio file and visualize the results:
```bash
python run.py inference --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt --audio_file path/to/audio.wav
```

Generate audio using the transformer model:
```bash
python run.py inference --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt --transformer_checkpoint checkpoints/transformer_epoch_100.pt --generate
```

## Project Structure

```
audio_transformer/
├── __init__.py          # Package initialization
├── data/                # Data processing scripts and datasets
├── models/              # Neural network models
│   ├── __init__.py      # Models package initialization
│   ├── transformer.py   # Transformer model implementations
│   └── vq_vae.py        # VQ-VAE model implementation
└── utils/               # Utility functions
    ├── __init__.py      # Utils package initialization
    └── audio_processing.py  # Audio processing utilities
demo.py                  # Demo script for visualization
requirements.txt         # Package dependencies
run.py                   # Main runner script
```

## Advanced Configuration

You can customize various parameters:

- Audio processing parameters:
  - `--sample_rate`: Audio sample rate (default: 22050)
  - `--n_fft`: FFT size (default: 1024)
  - `--hop_length`: Hop length (default: 512)
  - `--n_mels`: Number of mel bands (default: 80)

- Training parameters:
  - `--learning_rate`: Learning rate (default: 1e-4)
  - `--batch_size`: Batch size (default: 16)
  - `--num_epochs`: Number of training epochs (default: 100)
  - `--val_split`: Validation split ratio (default: 0.1)

- Generation parameters:
  - `--max_length`: Maximum sequence length for generation (default: 1000)
  - `--temperature`: Sampling temperature (default: 1.0)

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchaudio 0.9.0+
- librosa 0.8.1+
- Additional packages listed in requirements.txt 