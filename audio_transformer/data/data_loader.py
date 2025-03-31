import os
import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio
from typing import Dict, Tuple, List, Optional, Union

class AudioDataModule:
    """
    Data module for handling audio datasets using the datasets library
    """
    def __init__(
        self,
        dataset_type: str,
        audio_dir: Optional[str] = None,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        batch_size: int = 16,
        val_split: float = 0.1,
        num_workers: int = 4,
    ):
        self.dataset_type = dataset_type
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Set up the appropriate datasets based on dataset_type"""
        if self.dataset_type == "gtzan":
            self._setup_gtzan()
        elif self.dataset_type == "fsdd":
            self._setup_fsdd()
        elif self.dataset_type == "urbansound":
            self._setup_urbansound()
        elif self.dataset_type == "custom" and self.audio_dir:
            self._setup_custom()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Add audio processing pipeline
        self._add_audio_processing()
    
    def _setup_gtzan(self):
        """Setup the GTZAN dataset"""
        if self.audio_dir:
            # Use local GTZAN dataset if provided
            dataset = load_dataset(
                "audiofolder",
                data_dir=self.audio_dir,
                split="train"
            )
        else:
            # Use GTZAN dataset from Hugging Face
            dataset = load_dataset("marsyas/gtzan", split="train")
        
        # Filter out any problematic samples
        dataset = dataset.filter(lambda x: x['audio']['array'].shape[0] > 0)
        
        # Split into train/val/test
        splits = dataset.train_test_split(test_size=self.val_split * 2)
        train_dataset = splits['train']
        test_val = splits['test'].train_test_split(test_size=0.5)
        
        self.train_dataset = train_dataset
        self.val_dataset = test_val['train']
        self.test_dataset = test_val['test']
    
    def _setup_fsdd(self):
        """Setup the Free Spoken Digit Dataset"""
        if self.audio_dir:
            # Use local FSDD dataset if provided
            dataset = load_dataset(
                "audiofolder",
                data_dir=self.audio_dir,
                split="train"
            )
        else:
            # Use FSDD dataset from Hugging Face
            dataset = load_dataset("Robolab/FSDD", split="train")
        
        # Filter out any problematic samples
        dataset = dataset.filter(lambda x: x['audio']['array'].shape[0] > 0)
        
        # Split into train/val/test
        splits = dataset.train_test_split(test_size=self.val_split * 2)
        train_dataset = splits['train']
        test_val = splits['test'].train_test_split(test_size=0.5)
        
        self.train_dataset = train_dataset
        self.val_dataset = test_val['train']
        self.test_dataset = test_val['test']
    
    def _setup_urbansound(self):
        """Setup the UrbanSound8K dataset"""
        if self.audio_dir:
            # Use local UrbanSound8K dataset if provided
            dataset = load_dataset(
                "audiofolder",
                data_dir=self.audio_dir,
                split="train"
            )
        else:
            # Use UrbanSound8K dataset from Hugging Face
            dataset = load_dataset("ashraq/urbansound8k", split="train")
        
        # Filter out any problematic samples
        dataset = dataset.filter(lambda x: x['audio']['array'].shape[0] > 0)
        
        # Split into train/val/test
        splits = dataset.train_test_split(test_size=self.val_split * 2)
        train_dataset = splits['train']
        test_val = splits['test'].train_test_split(test_size=0.5)
        
        self.train_dataset = train_dataset
        self.val_dataset = test_val['train']
        self.test_dataset = test_val['test']
    
    def _setup_custom(self):
        """Setup a custom dataset from a directory"""
        if not self.audio_dir:
            raise ValueError("audio_dir must be provided for custom dataset")
        
        dataset = load_dataset(
            "audiofolder",
            data_dir=self.audio_dir,
            split="train"
        )
        
        # Filter out any problematic samples
        dataset = dataset.filter(lambda x: x['audio']['array'].shape[0] > 0)
        
        # Split into train/val/test
        splits = dataset.train_test_split(test_size=self.val_split * 2)
        train_dataset = splits['train']
        test_val = splits['test'].train_test_split(test_size=0.5)
        
        self.train_dataset = train_dataset
        self.val_dataset = test_val['train']
        self.test_dataset = test_val['test']
    
    def _add_audio_processing(self):
        """Add audio processing to the datasets"""
        def extract_melspectrogram(batch):
            """Convert audio to mel spectrogram"""
            audio = batch["audio"]["array"]
            sr = batch["audio"]["sampling_rate"]
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(
                    audio.astype(np.float32), 
                    orig_sr=sr, 
                    target_sr=self.sample_rate
                )
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.T)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / max(log_mel_spec.std(), 1e-8)
            
            batch["mel_spectrogram"] = log_mel_spec.astype(np.float32)
            return batch
        
        # Apply processing to all datasets
        if self.train_dataset:
            self.train_dataset = self.train_dataset.map(
                extract_melspectrogram,
                num_proc=self.num_workers
            )
        
        if self.val_dataset:
            self.val_dataset = self.val_dataset.map(
                extract_melspectrogram,
                num_proc=self.num_workers
            )
        
        if self.test_dataset:
            self.test_dataset = self.test_dataset.map(
                extract_melspectrogram,
                num_proc=self.num_workers
            )
    
    def _collate_fn(self, batch):
        """Custom collate function for the DataLoader"""
        mel_specs = [item["mel_spectrogram"] for item in batch]
        
        # Get max length
        max_length = max(spec.shape[1] for spec in mel_specs)
        
        # Pad all spectrograms to max length
        padded_specs = []
        for spec in mel_specs:
            pad_width = max_length - spec.shape[1]
            padded_spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
            padded_specs.append(padded_spec)
        
        # Convert to tensor
        mel_specs_tensor = torch.tensor(np.stack(padded_specs))
        
        # Get labels if available
        labels = None
        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch])
        
        return {
            "mel_spectrogram": mel_specs_tensor,
            "labels": labels
        }
    
    def train_dataloader(self):
        """Return the train dataloader"""
        if self.train_dataset is None:
            self.setup()
        
        return DataLoader(
            self.train_dataset.with_format("torch"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        """Return the validation dataloader"""
        if self.val_dataset is None:
            self.setup()
        
        return DataLoader(
            self.val_dataset.with_format("torch"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        """Return the test dataloader"""
        if self.test_dataset is None:
            self.setup()
        
        return DataLoader(
            self.test_dataset.with_format("torch"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        ) 