import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa

class AudioProcessor:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
    ):
        """
        Initialize audio processing pipeline
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for FFT
            n_mels: Number of Mel filter banks
            f_min: Minimum frequency for Mel filter
            f_max: Maximum frequency for Mel filter
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        # Initialize Mel spectrogram transformer
        self.mel_spec_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
    
    def load_audio(self, file_path, normalize=True):
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio to [-1, 1]
            
        Returns:
            waveform: Audio waveform
            sample_rate: Sample rate of loaded audio
        """
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if normalize:
            waveform = waveform / (torch.abs(waveform).max() + 1e-8)
            
        return waveform, self.sample_rate
    
    def convert_to_mono(self, waveform):
        """
        Convert multi-channel audio to mono
        
        Args:
            waveform: Audio waveform (channels, time)
            
        Returns:
            mono_waveform: Mono audio waveform (1, time)
        """
        if waveform.shape[0] > 1:
            # Average all channels
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            mono_waveform = waveform
            
        return mono_waveform
    
    def extract_mel_spectrogram(self, waveform):
        """
        Convert waveform to Mel spectrogram
        
        Args:
            waveform: Audio waveform (1, time)
            
        Returns:
            mel_spec: Mel spectrogram (1, n_mels, time)
        """
        # Apply Mel spectrogram transform
        mel_spec = self.mel_spec_transform(waveform)
        
        # Convert to log scale (dB) with small offset to avoid log(0)
        log_mel_spec = torch.log10(mel_spec + 1e-9)
        
        return log_mel_spec
    
    def extract_fft(self, waveform):
        """
        Extract FFT frames from waveform
        
        Args:
            waveform: Audio waveform (1, time)
            
        Returns:
            fft_frames: FFT frames (frames, n_fft//2+1)
        """
        # Convert to numpy for using librosa's frame function
        audio_np = waveform.squeeze().numpy()
        
        # Frame the signal
        frames = librosa.util.frame(audio_np, frame_length=self.n_fft, hop_length=self.hop_length)
        
        # Apply windowing
        frames = frames * np.hanning(self.n_fft)[:, np.newaxis]
        
        # Compute FFT
        fft_frames = np.abs(np.fft.rfft(frames, axis=0))
        
        return torch.from_numpy(fft_frames.T)
    
    def process_audio(self, file_path=None, waveform=None):
        """
        Full audio processing pipeline
        
        Args:
            file_path: Path to audio file (optional)
            waveform: Audio waveform (optional)
            
        Returns:
            mel_spec: Mel spectrogram features
        """
        if file_path is not None:
            waveform, _ = self.load_audio(file_path)
        
        if waveform is None:
            raise ValueError("Either file_path or waveform must be provided")
            
        # Convert to mono
        mono_waveform = self.convert_to_mono(waveform)
        
        # Extract Mel spectrogram
        mel_spec = self.extract_mel_spectrogram(mono_waveform)
        
        return mel_spec 