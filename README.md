# Audio Transformer

[English](README.md) | [中文](README_CN.md)

# Audio Conversion Transformer

A PyTorch implementation of an audio processing pipeline with Transformer models that converts audio through the following steps:

1. Multi-channel to mono conversion
2. Frame segmentation + FFT
3. Discrete spectrum extraction
4. Mel filtering
5. Discrete Mel spectrum (frequency domain discretization)
6. (Optional) VQ-VAE/clustering for more compact discrete tokens
7. Transformer input (sequence length = frame count, token dimension = feature dimension)

## Features

- Complete audio processing pipeline from raw audio to mel spectrograms
- Vector Quantized Variational Autoencoder (VQ-VAE) for discrete audio representation
- Transformer models for sequence modeling of audio features
- Support for both discrete token-based processing (with VQ-VAE) and continuous feature processing
- Built-in training and inference pipelines
- Integration with 🤗 Hugging Face's `datasets` and `transformers` libraries
- Visualization tools for audio analysis

## Installation

```bash
git clone https://github.com/username/audio-conv-transformer.git
cd audio-conv-transformer
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchaudio 0.9.0+
- librosa 0.8.1+
- datasets 2.10.0+
- transformers 4.26.0+
- Additional packages listed in requirements.txt 

## Default Datasets

For convenience, we recommend the following datasets for training and evaluation:

1. **GTZAN Genre Collection**:
   - 1000 audio tracks (30 seconds each)
   - 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
   - Size: ~1.2GB
   - [Download link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

2. **Free Spoken Digit Dataset (FSDD)**:
   - Smaller dataset ideal for testing
   - Spoken digits (0-9) with multiple speakers
   - Size: ~10MB
   - [GitHub Repository](https://github.com/Jakobovski/free-spoken-digit-dataset)

3. **UrbanSound8K**:
   - 8732 labeled sound excerpts (≤4s) of urban sounds
   - 10 classes
   - Size: ~6GB
   - [Download link](https://urbansounddataset.weebly.com/urbansound8k.html)

To use these datasets with the project:

```bash
# For GTZAN
python run.py train --audio_dir path/to/gtzan/genres --model_type vqvae --dataset_type gtzan

# For FSDD
python run.py train --audio_dir path/to/fsdd/recordings --model_type vqvae --dataset_type fsdd

# For UrbanSound8K
python run.py train --audio_dir path/to/urbansound8k/audio --model_type vqvae --dataset_type urbansound
```

The project includes built-in dataset loaders for these common datasets which handle the specific directory structures and metadata formats.

## Usage

### Training a VQ-VAE model

```bash
python run.py train \
  --model_type vqvae \
  --dataset_type gtzan \
  --audio_dir path/to/audio/files \
  --batch_size 32 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --output_dir ./checkpoints/vqvae
```

### Training a Transformer model (requires pre-trained VQ-VAE)

```bash
python run.py train \
  --model_type transformer \
  --dataset_type gtzan \
  --audio_dir path/to/audio/files \
  --vqvae_checkpoint ./checkpoints/vqvae \
  --batch_size 16 \
  --num_epochs 100 \
  --learning_rate 5e-5 \
  --output_dir ./checkpoints/transformer
```

### Inference with VQ-VAE and Transformer

```bash
python run.py inference \
  --vqvae_checkpoint ./checkpoints/vqvae \
  --transformer_checkpoint ./checkpoints/transformer \
  --audio_file path/to/audio.wav
```

### Demo Mode

For a quick demonstration, run:

```bash
python demo.py \
  --audio_file path/to/audio.wav \
  --vqvae_checkpoint ./checkpoints/vqvae
```

## Advanced Features

### Configuration Options

You can configure the audio processing parameters:

```bash
python run.py train \
  --model_type vqvae \
  --dataset_type custom \
  --audio_dir path/to/audio/files \
  --sample_rate 22050 \
  --n_fft 1024 \
  --hop_length 512 \
  --n_mels 80 \
  --codebook_size 8192 \
  --codebook_dim 64 \
  --hidden_dim 512
```

### Continuous Feature Transformer

To train a Transformer directly on continuous features (without VQ-VAE):

```bash
python run.py train \
  --model_type feature_transformer \
  --dataset_type custom \
  --audio_dir path/to/audio/files
```

## Project Structure

- **audio_transformer/**
  - **data/**: Data loading and processing modules
  - **models/**: Model definitions (VQ-VAE, Transformer)
  - **utils/**: Utility functions and training helpers
- **run.py**: Main script for training and inference
- **demo.py**: Demo script for quick testing

## Architecture Details

### VQ-VAE

The VQ-VAE consists of:
- An encoder that compresses the mel spectrogram into a latent space
- A vector quantization layer that maps continuous features to a discrete codebook
- A decoder that reconstructs the mel spectrogram from the quantized representation

### Transformer Models

Two types of transformer models are implemented:
1. **AudioTransformer**: Works with discrete tokens from VQ-VAE
2. **AudioFeatureTransformer**: Works directly with continuous mel spectrogram features

Both are based on Hugging Face's transformer architectures, using either BERT or Wav2Vec2 as the underlying model.

## Integration with 🤗 Hugging Face Libraries

This project leverages Hugging Face's ecosystem in several ways:

1. **Datasets**: Uses the `datasets` library for efficient data loading, caching, and preprocessing
   - Supports loading from local files or directly from the Hugging Face Hub
   - Built-in dataset loading for common audio datasets

2. **Transformers**: Models extend the `PreTrainedModel` class for seamless integration
   - Compatible with the Hugging Face model hub
   - Can use existing pretrained models as a base
   - Supports standard methods like `from_pretrained()` and `save_pretrained()`

3. **Training**: Uses the `Trainer` API for efficient training loops
   - Handles checkpointing, logging, and evaluation
   - Supports early stopping and other training optimizations

## License

[MIT License](LICENSE)

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [VQ-VAE paper](https://arxiv.org/abs/1711.00937)
- [PyTorch](https://pytorch.org/)
- [librosa](https://librosa.org/) 