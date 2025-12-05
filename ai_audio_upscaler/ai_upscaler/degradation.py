import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import math

class AdvancedDegradation(nn.Module):
    """
    Simulates real-world audio degradation to train robust restoration models.
    Features:
    - Reverb (Convolution with random noise decay)
    - EQ / Coloration (Random Biquad Filters)
    - Clipping (Soft/Hard)
    - Bandwidth Limiting (Low-pass)
    - Additive Noise (Gaussian/Colored)
    """
    def __init__(self, sample_rate=48000, max_cutoff=24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_cutoff = max_cutoff
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (Batch, Channels, Time) or (Channels, Time)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            
        # Apply degradations in random order or fixed pipeline?
        # Fixed pipeline is usually fine as long as probabilities are independent.
        
        # 1. Reverb (Simulate room tone)
        if random.random() < 0.3:
            waveform = self.apply_reverb(waveform)
            
        # 2. EQ / Coloration (Simulate bad mics)
        if random.random() < 0.3:
            waveform = self.apply_random_eq(waveform)
            
        # 3. Additive Noise (Simulate tape hiss / preamp noise)
        if random.random() < 0.3:
            waveform = self.apply_noise(waveform)
            
        # 4. Clipping (Simulate recording levels too hot)
        if random.random() < 0.2:
            waveform = self.apply_clipping(waveform)
            
        # 5. Bandwidth Limiting (Simulate low-quality source)
        # Always run this (or with high prob) to ensure we cover the frequency gap
        waveform = self.apply_bandwidth_limit(waveform)
        
        return waveform.squeeze(0) if waveform.shape[0] == 1 else waveform

    def apply_reverb(self, waveform):
        """Simple convolution reverb with exponential decay noise."""
        device = waveform.device
        # Random decay length (0.1s to 0.5s)
        reverb_len = int(random.uniform(0.1, 0.5) * self.sample_rate)
        if reverb_len == 0: return waveform
        
        # Create Impulse Response (White noise * exponential decay)
        t = torch.linspace(0, 1, reverb_len, device=device)
        decay = torch.exp(-t * random.uniform(5, 15)) # Decay rate
        ir = torch.randn(1, 1, reverb_len, device=device) * decay.view(1, 1, -1)
        
        # Normalize IR energy
        ir = ir / torch.norm(ir) * 0.5 # Mix level
        
        # Convolve
        # Pad waveform to avoid shortening
        padded = F.pad(waveform, (reverb_len-1, 0))
        wet = F.conv1d(padded, ir, groups=waveform.shape[1])
        
        # Mix Dry/Wet (Random mix)
        mix = random.uniform(0.1, 0.4)
        return (1 - mix) * waveform + mix * wet[..., :waveform.shape[-1]]

    def apply_random_eq(self, waveform):
        """Apply random biquad filter (Peaking/Notch) to simulate coloration."""
        # Torchaudio biquad requires CPU processing usually, or we can use lfilter
        # For simplicity and GPU speed, we can use a simple 3-band shelf approximation 
        # or just torchaudio.functional.equalizer_biquad if it supports CUDA (it does).
        
        center_freq = random.uniform(200, 8000)
        gain = random.uniform(-10, 10) # dB
        q = random.uniform(0.5, 2.0)
        
        return torchaudio.functional.equalizer_biquad(
            waveform, 
            self.sample_rate, 
            center_freq, 
            gain, 
            q
        )

    def apply_clipping(self, waveform):
        """Soft or Hard clipping."""
        mode = random.choice(["hard", "soft"])
        threshold = random.uniform(0.5, 0.9)
        
        if mode == "hard":
            return torch.clamp(waveform, -threshold, threshold)
        else:
            # Soft clipping (tanh-like)
            # f(x) = tanh(x / t) * t
            return torch.tanh(waveform / threshold) * threshold

    def apply_noise(self, waveform):
        """Add Gaussian noise."""
        snr_db = random.uniform(20, 50)
        noise = torch.randn_like(waveform)
        
        # Calculate signal power
        sig_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        # Scale noise to target SNR
        scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
        return waveform + noise * scale

    def apply_bandwidth_limit(self, waveform):
        """Resampling-based low-pass filter."""
        # Random cutoff between 4kHz and Nyquist
        cutoff = random.randint(4000, self.max_cutoff)
        
        # If cutoff is near Nyquist, skip (simulate perfect audio)
        if cutoff >= self.sample_rate // 2 * 0.95:
            return waveform
            
        down_sr = cutoff * 2
        
        # Resample down and up
        # Note: Creating resamplers on the fly is slow. 
        # For training, it's better to cache or use a functional form if available.
        # torchaudio.transforms.Resample is efficient if reused, but here parameters change.
        # We'll use functional resampling with linear interpolation for speed/simplicity 
        # or sinc if we want quality. Linear is actually better for simulating "bad" upsampling.
        
        # Simple interpolation
        orig_len = waveform.shape[-1]
        down = F.interpolate(waveform, scale_factor=down_sr/self.sample_rate, mode='linear', align_corners=False)
        up = F.interpolate(down, size=orig_len, mode='linear', align_corners=False)
        
        return up
