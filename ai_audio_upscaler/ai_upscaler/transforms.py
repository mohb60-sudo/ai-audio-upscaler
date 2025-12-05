import torch
import torchaudio
import io
import random
import logging

logger = logging.getLogger(__name__)

class MP3Compression:
    """
    Apply MP3 compression artifacts to the audio.
    """
    def __init__(self, min_bitrate=64, max_bitrate=192, sample_rate=44100):
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.sample_rate = sample_rate
        
    def __call__(self, waveform):
        """
        Args:
            waveform: (Channels, Time)
        """
        # Randomly skip compression sometimes
        if random.random() < 0.2:
            return waveform

        # Pick a random bitrate
        # Standard MP3 bitrates: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320
        # We'll approximate by picking a value in range
        # bitrate = random.choice([64, 80, 96, 112, 128, 160, 192]) (Unused)
        
        # Torchaudio save expects (Channels, Time)
        # We need to save to a buffer
        buffer = io.BytesIO()
        
        try:
            # Ensure waveform is on CPU for saving
            w_cpu = waveform.cpu()
            
            # MP3 only supports up to 48kHz. If we are at 96kHz, we must downsample first.
            encoding_sr = self.sample_rate
            if encoding_sr > 48000:
                encoding_sr = 48000
                resampler_down = torchaudio.transforms.Resample(self.sample_rate, encoding_sr)
                w_cpu = resampler_down(w_cpu)
            
            # Save as MP3 (Default bitrate since soundfile backend doesn't support compression arg)
            torchaudio.save(buffer, w_cpu, encoding_sr, format="mp3")
            
            # Load back
            buffer.seek(0)
            loaded_w, loaded_sr = torchaudio.load(buffer, format="mp3")
            
            # If we downsampled, we must upsample back to original rate
            if loaded_sr != self.sample_rate:
                resampler_up = torchaudio.transforms.Resample(loaded_sr, self.sample_rate)
                loaded_w = resampler_up(loaded_w)
            
            # Ensure length matches (MP3 encoding can add padding)
            if loaded_w.shape[1] != waveform.shape[1]:
                min_len = min(loaded_w.shape[1], waveform.shape[1])
                loaded_w = loaded_w[:, :min_len]
                # If loaded is shorter (rare but possible), pad it? 
                # Usually MP3 adds silence at start/end.
                if loaded_w.shape[1] < waveform.shape[1]:
                     loaded_w = torch.nn.functional.pad(loaded_w, (0, waveform.shape[1] - loaded_w.shape[1]))
            
            return loaded_w.to(waveform.device)
            
        except Exception as e:
            logger.warning(f"MP3 Compression failed: {e}")
            return waveform

class QuantizationNoise:
    """
    Simulate lower bit depth (quantization noise).
    """
    def __init__(self, min_bits=8, max_bits=14):
        self.min_bits = min_bits
        self.max_bits = max_bits
        
    def __call__(self, waveform):
        if random.random() < 0.2:
            return waveform
            
        bits = random.randint(self.min_bits, self.max_bits)
        
        # Quantize
        q_levels = 2 ** bits
        
        # Assume waveform is roughly -1 to 1
        waveform_q = torch.round(waveform * (q_levels / 2)) / (q_levels / 2)
        
        return waveform_q

class BandwidthLimiter:
    """
    Apply a low-pass filter to simulate bandwidth limitation.
    """
    def __init__(self, sample_rate, min_cutoff=4000, max_cutoff=16000):
        self.sample_rate = sample_rate
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        
    def __call__(self, waveform):
        if random.random() < 0.2:
            return waveform
            
        cutoff = random.randint(self.min_cutoff, self.max_cutoff)
        
        # Simple implementation using resampling as a low-pass filter
        # Downsample to 2*cutoff then upsample back
        # This effectively removes frequencies above cutoff
        
        down_sr = cutoff * 2
        
        # Don't upsample if cutoff is higher than Nyquist
        if down_sr >= self.sample_rate:
            return waveform
            
        resampler_down = torchaudio.transforms.Resample(self.sample_rate, down_sr)
        resampler_up = torchaudio.transforms.Resample(down_sr, self.sample_rate)
        
        down = resampler_down(waveform)
        up = resampler_up(down)
        
        # Match length
        if up.shape[1] != waveform.shape[1]:
             min_len = min(up.shape[1], waveform.shape[1])
             up = up[:, :min_len]
             if up.shape[1] < waveform.shape[1]:
                 up = torch.nn.functional.pad(up, (0, waveform.shape[1] - up.shape[1]))
                 
        return up
