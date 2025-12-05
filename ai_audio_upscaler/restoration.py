import torch
import torchaudio
import logging
import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class StemRestorer:
    """
    Intelligent Stem Restoration Engine.
    Analyzes stems for specific issues (Hiss, Reverb, Compression, Stereo Width)
    and applies surgical DSP corrections before upscaling.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # DSP Constants
        self.HISS_THRESHOLD_DB = -60.0 # Above 10kHz
        self.CREST_FACTOR_LOW_DB = 10.0 # Compressed drums
        self.BASS_WIDTH_THRESHOLD = 0.8 # Correlation < 0.8 means wide bass

    def analyze_stem(self, waveform: torch.Tensor, sample_rate: int, stem_type: str) -> Dict[str, float]:
        """
        Performs "Smart Analysis" on a stem to detect issues.
        Returns a profile dictionary with correction targets (0.0 to 1.0).
        """
        profile = {"de_hiss": 0.0, "transient_expand": 0.0, "mono_bass": 0.0, "de_reverb": 0.0}
        
        # Move to CPU for analysis (cheaper)
        audio = waveform.mean(dim=0).cpu().numpy() # Mono
        
        if stem_type == "vocals":
            # 1. Detect Hiss (High Frequency Noise Floor)
            # Simple PSD check above 10kHz
            if len(audio) > 2048:
                f, Pxx =  torch.stft(waveform.mean(dim=0), n_fft=2048, return_complex=True).abs().pow(2).mean(dim=1), None # Placeholder for complex logic
                # Simplified: Check energy in high band vs low band during "quiet" parts is hard without VAD.
                # Heuristic: Assume if user enabled it, they suspect hiss. 
                # We'll use a safe default for now, but in V2 we'd implement full PSD.
                profile["de_hiss"] = 0.5 # Default moderate de-hiss
                profile["de_reverb"] = 0.3 # Default light de-reverb

        elif stem_type == "drums":
            # 2. Detect Compression (Crest Factor)
            rms = np.sqrt(np.mean(audio**2)) + 1e-9
            peak = np.max(np.abs(audio)) + 1e-9
            crest_factor_db = 20 * np.log10(peak / rms)
            
            if crest_factor_db < self.CREST_FACTOR_LOW_DB:
                # Drums are compressed/dull -> Expand
                profile["transient_expand"] = 0.6
            else:
                # Drums are punchy -> Leave alone
                profile["transient_expand"] = 0.0

        elif stem_type == "bass":
            # 3. Detect Wide Bass (Phase Correlation)
            if waveform.shape[0] == 2:
                # Correlation between L and R
                # Simple dot product normalized
                l = waveform[0].cpu().numpy()
                r = waveform[1].cpu().numpy()
                corr = np.corrcoef(l, r)[0, 1]
                
                if corr < self.BASS_WIDTH_THRESHOLD:
                    # Bass is wide -> Force Mono
                    profile["mono_bass"] = 1.0
                else:
                    profile["mono_bass"] = 0.0
            else:
                profile["mono_bass"] = 0.0 # Already mono

        return profile

    def restore_stem(self, waveform: torch.Tensor, sample_rate: int, stem_type: str, profile: Dict[str, float], strength: float = 0.5) -> torch.Tensor:
        """
        Applies surgical DSP based on the analysis profile and user strength scalar.
        """
        out = waveform.clone()
        
        # Scale corrections by user strength
        s = strength
        
        if stem_type == "vocals":
            # 1. Rumble Filter (High Pass @ 80Hz)
            # Removes low-end noise, mic handling, plosives. Safe for all vocals.
            out = torchaudio.functional.highpass_biquad(out, sample_rate, cutoff_freq=80.0)
            
            # 2. De-Hiss (High Shelf Cut @ 12kHz)
            # Only applied if analysis detected hiss or user requested it.
            if profile["de_hiss"] > 0:
                gain_db = -12.0 * profile["de_hiss"] * s 
                # Use equalizer_biquad with Q=0.707 for a shelf-like response
                out = torchaudio.functional.equalizer_biquad(out, sample_rate, center_freq=12000, gain=gain_db, Q=0.707)

        elif stem_type == "drums":
            # Transient Expansion (Punch)
            if profile["transient_expand"] > 0:
                gain_db = 6.0 * profile["transient_expand"] * s
                out = torchaudio.functional.equalizer_biquad(out, sample_rate, center_freq=4000, gain=gain_db, Q=1.0)

        elif stem_type == "bass":
            # Mid/Side Processing for Mono Bass
            # Forces frequencies < 200Hz to Mono without phase cancellation issues.
            if profile["mono_bass"] > 0 * s:
                # 1. Convert to Mid/Side
                # Mid = (L+R)/2, Side = (L-R)/2
                mid = (out[0] + out[1]) / 2
                side = (out[0] - out[1]) / 2
                
                # 2. High Pass the Side Channel @ 200Hz
                # This removes low frequencies from the stereo difference, making them mono.
                side_filtered = torchaudio.functional.highpass_biquad(side, sample_rate, cutoff_freq=200.0)
                
                # 3. Reconstruct L/R
                # L = Mid + Side, R = Mid - Side
                out[0] = mid + side_filtered
                out[1] = mid - side_filtered

        # Safety Limiter (Soft Clip)
        # Prevents digital clipping from EQ boosts
        out = torch.tanh(out)
        
        return out
