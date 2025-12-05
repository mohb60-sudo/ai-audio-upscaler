import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt
from typing import Tuple

class AudioPreProcessor:
    """
    Handles input conditioning before AI processing.
    Includes DC offset removal, LUFS normalization, and de-clicking.
    """
    
    @staticmethod
    def remove_dc_offset(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Removes DC offset using a high-pass filter at 10Hz.
        
        Args:
            waveform (torch.Tensor): Audio data (Channels, Time).
            sr (int): Sample rate.
            
        Returns:
            torch.Tensor: Filtered waveform.
        """
        # Convert to numpy for scipy processing
        wav_np = waveform.cpu().numpy()
        
        # Design 10Hz Highpass
        sos = butter(2, 10, 'hp', fs=sr, output='sos')
        
        # Apply filter
        filtered = sosfilt(sos, wav_np, axis=-1)
        
        return torch.from_numpy(filtered.copy()).float()

    @staticmethod
    def normalize_lufs(waveform: torch.Tensor, sr: int, target_lufs: float = -18.0) -> Tuple[torch.Tensor, float]:
        """
        Normalizes audio to target LUFS.
        
        Args:
            waveform (torch.Tensor): Audio data.
            sr (int): Sample rate.
            target_lufs (float): Target Integrated Loudness in LUFS.
            
        Returns:
            Tuple[torch.Tensor, float]: (normalized_waveform, gain_applied_db)
        """
        meter = pyln.Meter(sr)
        
        # pyloudnorm expects (samples, channels)
        wav_np = waveform.cpu().t().numpy()
        
        try:
            loudness = meter.integrated_loudness(wav_np)
        except ValueError:
            # Silence or too short
            return waveform, 0.0
            
        if np.isinf(loudness):
            return waveform, 0.0
            
        delta_lufs = target_lufs - loudness
        gain_linear = 10.0 ** (delta_lufs / 20.0)
        
        # Apply gain
        normalized = waveform * gain_linear
        
        return normalized, delta_lufs

    @staticmethod
    def de_click(waveform: torch.Tensor) -> torch.Tensor:
        """
        Simple statistical de-clicker.
        Clamps samples that are > 10 std devs from local mean.
        
        Args:
            waveform (torch.Tensor): Input audio.
            
        Returns:
            torch.Tensor: De-clicked audio.
        """
        # This is a very basic implementation. 
        # For professional de-clicking we'd need a heavier algorithm.
        # We'll use a simple clamp for now to catch digital glitches.
        
        std = torch.std(waveform)
        mean = torch.mean(waveform)
        threshold = 10 * std
        
        return torch.clamp(waveform, min=mean - threshold, max=mean + threshold)
