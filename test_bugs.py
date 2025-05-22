import unittest
import torch
import numpy as np
import os
from audio_transformer.utils.audio_processing import AudioProcessor, load_and_process_audio, mel_to_audio
from audio_transformer.models.vq_vae import VQVAE
from audio_transformer.models.transformer import AudioTransformer, AudioFeatureTransformer
from audio_transformer.data.data_loader import AudioDataModule
import tempfile

class TestAudioProcessing(unittest.TestCase):
    def test_audio_processor_initialization(self):
        # Test initialization with default parameters
        processor = AudioProcessor()
        self.assertEqual(processor.sample_rate, 22050)
        self.assertEqual(processor.n_fft, 1024)
        self.assertEqual(processor.hop_length, 512)
        self.assertEqual(processor.n_mels, 80)
        self.assertTrue(processor.normalize)

    def test_extract_mel_spectrogram(self):
        # Create a dummy audio signal
        sample_rate = 22050
        duration = 1  # seconds
        audio = np.sin(2 * np.pi * 440 * np.arange(duration * sample_rate) / sample_rate)
        
        processor = AudioProcessor(sample_rate=sample_rate)
        mel_spec = processor.extract_mel_spectrogram(audio)
        
        # Check shape and type
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertEqual(mel_spec.shape[0], processor.n_mels)
        self.assertGreater(mel_spec.shape[1], 0)

class TestVQVAE(unittest.TestCase):
    def test_vqvae_initialization(self):
        # Test model initialization
        try:
            model = VQVAE(
                in_channels=80,
                hidden_dims=[128, 256],
                embedding_dim=64,
                num_embeddings=512
            )
            self.assertIsInstance(model, VQVAE)
        except Exception as e:
            self.fail(f"VQVAE initialization failed: {e}")

    def test_vqvae_forward_pass(self):
        # Create a random mel spectrogram
        batch_size = 2
        n_mels = 80
        time_steps = 100
        mel_spec = torch.randn(batch_size, n_mels, time_steps)
        
        model = VQVAE(
            in_channels=n_mels,
            hidden_dims=[128, 256],
            embedding_dim=64,
            num_embeddings=512
        )
        
        # Test forward pass
        try:
            reconstructed, vq_loss, encoding_indices = model(mel_spec)
            
            # Check output shape
            self.assertEqual(reconstructed.shape, mel_spec.shape)
            self.assertIsInstance(vq_loss, torch.Tensor)
            self.assertEqual(encoding_indices.shape[0], batch_size)
        except Exception as e:
            self.fail(f"VQVAE forward pass failed: {e}")

class TestTransformer(unittest.TestCase):
    def test_transformer_initialization(self):
        # Test model initialization
        try:
            model = AudioTransformer(
                vocab_size=512,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6
            )
            self.assertIsInstance(model, AudioTransformer)
        except Exception as e:
            self.fail(f"AudioTransformer initialization failed: {e}")

    def test_feature_transformer_initialization(self):
        # Test model initialization
        try:
            model = AudioFeatureTransformer(
                input_dim=80,
                d_model=512,
                nhead=8,
                num_layers=6
            )
            self.assertIsInstance(model, AudioFeatureTransformer)
        except Exception as e:
            self.fail(f"AudioFeatureTransformer initialization failed: {e}")

class TestDataLoader(unittest.TestCase):
    def test_data_module_initialization(self):
        # Test initialization
        try:
            data_module = AudioDataModule(
                dataset_type="custom",
                audio_dir="/tmp",  # Just for testing initialization
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=80
            )
            self.assertIsInstance(data_module, AudioDataModule)
        except Exception as e:
            self.fail(f"AudioDataModule initialization failed: {e}")
            
if __name__ == '__main__':
    unittest.main()