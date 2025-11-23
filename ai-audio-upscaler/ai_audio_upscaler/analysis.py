import torch
import matplotlib.pyplot as plt
import numpy as np
import io

def generate_spectrogram(waveform: torch.Tensor, sample_rate: int, title: str = "Spectrogram"):
    """
    Generates a spectrogram plot for the given waveform.
    
    Args:
        waveform: Audio tensor (Channels, Time). We'll use the first channel.
        sample_rate: Sampling rate in Hz.
        title: Plot title.
        
    Returns:
        matplotlib Figure object.
    """
    # Convert to numpy and take first channel
    if waveform.dim() > 1:
        y = waveform[0].numpy()
    else:
        y = waveform.numpy()
        
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create spectrogram
    # NFFT=2048, Fs=sample_rate, noverlap=1024
    Pxx, freqs, bins, im = ax.specgram(y, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='inferno')
    
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig
