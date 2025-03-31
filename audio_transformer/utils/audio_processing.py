import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union, Any


def load_and_process_audio(
    file_path: str,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load audio file and convert to mel spectrogram
    
    Args:
        file_path: Path to audio file
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for FFT
        n_mels: Number of mel bands
        normalize: Whether to normalize the mel spectrogram
    
    Returns:
        Tuple of audio waveform and mel spectrogram
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize if requested
    if normalize:
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / max(log_mel_spec.std(), 1e-8)
    
    return audio, log_mel_spec


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    title: str = "Mel Spectrogram",
    save_path: Optional[str] = None
) -> None:
    """
    Plot mel spectrogram
    
    Args:
        mel_spec: Mel spectrogram array
        title: Plot title
        save_path: Path to save the figure (if None, will not save)
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 22050,
    title: str = "Waveform",
    save_path: Optional[str] = None
) -> None:
    """
    Plot audio waveform
    
    Args:
        audio: Audio waveform array
        sample_rate: Sample rate of the audio
        title: Plot title
        save_path: Path to save the figure (if None, will not save)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio)) / sample_rate, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def mel_to_audio(
    mel_spec: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 1.0,
    is_db_scale: bool = True,
    n_iter: int = 32
) -> np.ndarray:
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm
    
    Args:
        mel_spec: Mel spectrogram array
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for FFT
        power: Power for mel spectrogram
        is_db_scale: Whether the spectrogram is in dB scale
        n_iter: Number of iterations for Griffin-Lim
    
    Returns:
        Reconstructed audio
    """
    # Convert from dB scale if needed
    if is_db_scale:
        mel_spec = librosa.db_to_power(mel_spec)
    
    # Convert mel spectrogram to audio
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        n_iter=n_iter
    )
    
    return audio


def augment_audio(
    audio: np.ndarray,
    sample_rate: int = 22050,
    pitch_shift: Optional[int] = None,
    time_stretch: Optional[float] = None,
    add_noise: Optional[float] = None
) -> np.ndarray:
    """
    Apply augmentations to audio
    
    Args:
        audio: Audio waveform array
        sample_rate: Sample rate of the audio
        pitch_shift: Number of semitones to shift (+ up, - down)
        time_stretch: Stretch factor (>1 = slower, <1 = faster)
        add_noise: Signal-to-noise ratio in dB
    
    Returns:
        Augmented audio
    """
    # Make a copy of the audio to avoid modifying the original
    augmented_audio = audio.copy()
    
    # Apply pitch shift
    if pitch_shift is not None:
        augmented_audio = librosa.effects.pitch_shift(
            augmented_audio,
            sr=sample_rate,
            n_steps=pitch_shift
        )
    
    # Apply time stretch
    if time_stretch is not None:
        augmented_audio = librosa.effects.time_stretch(
            augmented_audio,
            rate=time_stretch
        )
    
    # Add noise
    if add_noise is not None:
        # Calculate signal power
        signal_power = np.mean(augmented_audio ** 2)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (add_noise / 10))
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(augmented_audio))
        
        # Add noise to signal
        augmented_audio = augmented_audio + noise
    
    return augmented_audio


def create_mel_dataset(
    audio_files: List[str],
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    normalize: bool = True,
    segment_length: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Create a dataset of mel spectrograms from a list of audio files
    
    Args:
        audio_files: List of paths to audio files
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for FFT
        n_mels: Number of mel bands
        normalize: Whether to normalize the mel spectrogram
        segment_length: Length of segments in frames (if None, use full audio)
    
    Returns:
        Dictionary of mel spectrograms and metadata
    """
    mel_specs = []
    audio_paths = []
    segments = []
    segment_idx = 0
    
    for audio_file in audio_files:
        # Load and process audio
        audio, mel_spec = load_and_process_audio(
            audio_file,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalize=normalize
        )
        
        # If segmenting
        if segment_length is not None:
            # Calculate number of segments
            num_frames = mel_spec.shape[1]
            num_segments = num_frames // segment_length
            
            # Create segments
            for i in range(num_segments):
                start_frame = i * segment_length
                end_frame = start_frame + segment_length
                
                # Extract segment
                segment = mel_spec[:, start_frame:end_frame]
                
                # Add to datasets
                mel_specs.append(segment)
                audio_paths.append(audio_file)
                segments.append({
                    "file_idx": len(audio_paths) - 1,
                    "segment_idx": segment_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame
                })
                segment_idx += 1
        else:
            # Use full spectrogram
            mel_specs.append(mel_spec)
            audio_paths.append(audio_file)
    
    return {
        "mel_specs": np.array(mel_specs),
        "audio_paths": audio_paths,
        "segments": segments if segment_length is not None else None
    }


class AudioProcessor:
    """
    Audio processor class for handling audio files
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono"""
        if len(audio.shape) > 1:
            return librosa.to_mono(audio)
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Normalize if requested
        if self.normalize:
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / max(log_mel_spec.std(), 1e-8)
        
        return log_mel_spec
    
    def extract_fft(self, audio: np.ndarray) -> np.ndarray:
        """Extract FFT from audio"""
        # Short-time Fourier transform
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to magnitude spectrogram
        mag_spec = np.abs(stft)
        
        return mag_spec
    
    def process_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Process audio file"""
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Convert to mono
        audio = self.convert_to_mono(audio)
        
        # Extract mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Extract FFT
        fft = self.extract_fft(audio)
        
        return {
            "audio": audio,
            "mel_spectrogram": mel_spec,
            "fft": fft,
            "sample_rate": sr
        } 