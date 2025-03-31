import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import logging

from ..models.vq_vae import VQVAE
from ..models.transformer import AudioTransformer, AudioFeatureTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VQVAETrainer(Trainer):
    """
    Custom trainer for VQVAE model that extends Hugging Face Trainer
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to handle VQVAE specific losses
        """
        outputs = model(features=inputs["mel_spectrogram"])
        
        # Get total loss from model outputs
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log to include VQVAE specific metrics
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class AudioTransformerTrainer(Trainer):
    """
    Custom trainer for AudioTransformer model that extends Hugging Face Trainer
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to handle AudioTransformer specific losses
        """
        # Prepare labels (shifted by 1 position for autoregressive training)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Get loss from model outputs
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


class VQVAECollator:
    """
    Data collator for VQVAE training
    """
    def __call__(self, examples):
        # Extract mel spectrograms
        mel_specs = [example["mel_spectrogram"] for example in examples]
        
        # Pad to max length
        max_length = max(mel.shape[1] for mel in mel_specs)
        padded_mels = []
        
        for mel in mel_specs:
            pad_length = max_length - mel.shape[1]
            padded_mel = np.pad(mel, ((0, 0), (0, pad_length)), mode='constant')
            padded_mels.append(padded_mel)
        
        # Convert to tensor
        batch = {
            "mel_spectrogram": torch.tensor(np.stack(padded_mels)).float(),
        }
        
        # Add labels if available
        if "label" in examples[0]:
            batch["labels"] = torch.tensor([example["label"] for example in examples])
        
        return batch


class AudioTransformerCollator:
    """
    Data collator for AudioTransformer training
    """
    def __init__(self, vqvae=None, use_vq=True, feature_dim=80):
        self.vqvae = vqvae
        self.use_vq = use_vq
        self.feature_dim = feature_dim
    
    def __call__(self, examples):
        # Extract mel spectrograms
        mel_specs = [example["mel_spectrogram"] for example in examples]
        
        # Pad to max length
        max_length = max(mel.shape[1] for mel in mel_specs)
        padded_mels = []
        
        for mel in mel_specs:
            pad_length = max_length - mel.shape[1]
            padded_mel = np.pad(mel, ((0, 0), (0, pad_length)), mode='constant')
            padded_mels.append(padded_mel)
        
        # Convert to tensor
        mel_specs_tensor = torch.tensor(np.stack(padded_mels)).float()
        
        # If using VQ-VAE, convert to discrete tokens
        if self.use_vq and self.vqvae is not None:
            with torch.no_grad():
                indices = self.vqvae.encode_to_indices(mel_specs_tensor)
            batch = {
                "input_ids": indices,
                "mel_spectrogram": mel_specs_tensor
            }
        else:
            # For continuous feature transformer
            batch = {
                "features": mel_specs_tensor.transpose(1, 2),  # [B, T, C]
                "mel_spectrogram": mel_specs_tensor
            }
        
        # Add labels if available
        if "label" in examples[0]:
            batch["labels"] = torch.tensor([example["label"] for example in examples])
        
        return batch


def train_vqvae(
    data_module,
    model_config,
    training_args,
    output_dir="./checkpoints",
    num_train_epochs=100,
    learning_rate=1e-4,
    batch_size=16,
    save_steps=1000,
    save_total_limit=5,
    logging_steps=100,
    eval_steps=500
):
    """
    Train VQVAE model using Hugging Face Trainer
    
    Args:
        data_module: Data module with train and val dataloaders
        model_config: Configuration for VQVAE model
        training_args: Training arguments or dict
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        save_steps: Save checkpoint every X steps
        save_total_limit: Maximum number of checkpoints to keep
        logging_steps: Log metrics every X steps
        eval_steps: Evaluate every X steps
    
    Returns:
        Trained VQVAE model
    """
    # Create model
    model = VQVAE(model_config)
    
    # Create data collator
    data_collator = VQVAECollator()
    
    # Create training arguments if not provided
    if not isinstance(training_args, TrainingArguments):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )
    
    # Create trainer
    trainer = VQVAETrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    return model


def train_audio_transformer(
    data_module,
    model_config,
    vqvae_model=None,
    training_args=None,
    output_dir="./checkpoints",
    num_train_epochs=100,
    learning_rate=5e-5,
    batch_size=16,
    save_steps=1000,
    save_total_limit=5,
    logging_steps=100,
    eval_steps=500,
    use_vq=True
):
    """
    Train AudioTransformer model using Hugging Face Trainer
    
    Args:
        data_module: Data module with train and val dataloaders
        model_config: Configuration for AudioTransformer model
        vqvae_model: Trained VQVAE model (if use_vq=True)
        training_args: Training arguments or dict
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        save_steps: Save checkpoint every X steps
        save_total_limit: Maximum number of checkpoints to keep
        logging_steps: Log metrics every X steps
        eval_steps: Evaluate every X steps
        use_vq: Whether to use VQ-VAE for discrete tokens
    
    Returns:
        Trained AudioTransformer model
    """
    # Create model based on whether we're using VQ or not
    if use_vq:
        model = AudioTransformer(model_config)
        if vqvae_model is None:
            raise ValueError("vqvae_model must be provided when use_vq=True")
    else:
        model = AudioFeatureTransformer(model_config)
    
    # Create data collator
    data_collator = AudioTransformerCollator(
        vqvae=vqvae_model,
        use_vq=use_vq,
        feature_dim=model_config.feature_dim if not use_vq else None
    )
    
    # Create training arguments if not provided
    if not isinstance(training_args, TrainingArguments):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            gradient_accumulation_steps=2,
            fp16=True,
        )
    
    # Create trainer
    trainer = AudioTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    return model 