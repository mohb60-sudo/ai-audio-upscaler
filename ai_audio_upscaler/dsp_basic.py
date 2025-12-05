import torch
import torchaudio
import logging

logger = logging.getLogger(__name__)

class DSPUpscaler:
    """
    Baseline DSP upscaling using standard resampling methods.
    """
    def __init__(self, target_sample_rate: int, method: str = "sinc") -> None:
        """
        Args:
            target_sample_rate: The desired output sample rate.
            method: 'sinc' (high quality) or 'linear' (fast).
        """
        self.target_sample_rate = target_sample_rate
        self.method = method

    def process(self, waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
        """
        Resample the waveform to the target sample rate.

        Args:
            waveform: Audio tensor of shape (channels, time).
            original_sample_rate: The sample rate of the input waveform.

        Returns:
            Resampled waveform tensor.
        """
        if original_sample_rate == self.target_sample_rate:
            logger.info("Original and target sample rates are identical. Skipping resampling.")
            return waveform

        logger.info(f"Resampling from {original_sample_rate} Hz to {self.target_sample_rate} Hz using {self.method} interpolation.")

        if self.method == "sinc":
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=self.target_sample_rate,
                resampling_method="sinc_interp_hann"
            )
            return resampler(waveform)
        
        elif self.method == "linear":
            # Linear interpolation requires 3D input (batch, channels, time) for torch.nn.functional.interpolate
            # We treat channels as batch for simplicity or unsqueeze
            # But torchaudio Resample doesn't support 'linear' directly in the same class easily with same API for all versions.
            # We'll use torch.nn.functional.interpolate for linear.
            
            # Shape: (Channels, Time) -> (1, Channels, Time)
            w_unsqueezed = waveform.unsqueeze(0) 
            
            # Calculate scale factor
            scale_factor = self.target_sample_rate / original_sample_rate
            
            out = torch.nn.functional.interpolate(
                w_unsqueezed, 
                scale_factor=scale_factor, 
                mode='linear', 
                align_corners=False
            )
            
            return out.squeeze(0)
        
        else:
            raise ValueError(f"Unknown resampling method: {self.method}")
