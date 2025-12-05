import torch
import numpy as np
from pedalboard import Pedalboard, Limiter, Compressor, HighpassFilter, Distortion, Gain
from pedalboard.io import AudioFile
import pyloudnorm as pyln

class AudioMasteringChain:
    """
    Handles post-processing mastering chain using Pedalboard.
    """
    
    def __init__(self, sr: int):
        self.sr = sr
        
    def process(self, waveform: torch.Tensor, 
                limit: bool = True, 
                target_lufs: float = -14.0, 
                saturation: float = 0.0,
                stereo_width: float = 1.0,
                transient_shaping: float = 0.0,
                dither: bool = False) -> torch.Tensor:
        """
        Runs the mastering chain on the input waveform.
        
        Args:
            waveform (torch.Tensor): Input audio (Channels, Time).
            limit (bool): Enable look-ahead limiter.
            target_lufs (float): Target loudness (approximate via gain).
            saturation (float): Amount of analog saturation (0.0 to 1.0).
            stereo_width (float): Stereo width multiplier (1.0 = original, >1.0 = wider).
            transient_shaping (float): Amount of transient punch (0.0 to 1.0).
            dither (bool): Apply TPDF dither for 24-bit output.
            
        Returns:
            torch.Tensor: Processed waveform.
        """
        # Convert to numpy (Channels, Time)
        audio_np = waveform.cpu().numpy()
        
        # Create a fresh board for this run (Thread-safe)
        board = Pedalboard()
        
        # 1. Saturation (Analog Warmth)
        if saturation > 0:
            # Use Distortion plugin as a saturator (soft clipping)
            drive_db = saturation * 20.0 # 0-1 -> 0-20dB drive
            board.append(Distortion(drive_db=drive_db))
            
        # 2. Transient Shaping / Dynamics
        if transient_shaping > 0:
            # "Punch" setting: Slow attack to let transients through
            board.append(Compressor(threshold_db=-10, ratio=2, attack_ms=30, release_ms=100))
            
        # 3. Limiter (True Peak)
        if limit:
            # Apply makeup gain to target roughly -14 LUFS from -18 LUFS input
            board.append(Gain(gain_db=4.0))
            board.append(Limiter(threshold_db=-1.0, release_ms=100.0))
            
        # Run processing
        # Pedalboard expects (Channels, Samples)
        processed = board(audio_np, self.sr)
        
        # 4. Manual Stereo Widening (Numpy)
        if stereo_width != 1.0 and processed.shape[0] == 2:
            mid = (processed[0] + processed[1]) * 0.5
            side = (processed[0] - processed[1]) * 0.5
            
            side *= stereo_width
            
            processed[0] = mid + side
            processed[1] = mid - side
            
        # 5. Dither (TPDF for 24-bit)
        if dither:
            # Triangular Probability Density Function (TPDF) Dither
            # Noise amplitude for 24-bit: +/- 1 LSB
            lsb_depth = 24
            scale = 1.0 / (2 ** (lsb_depth - 1))
            
            # Generate uniform noise [-1, 1] * scale
            noise = np.random.uniform(-1.0, 1.0, processed.shape) * scale
            # TPDF is sum of two uniform distributions
            noise += np.random.uniform(-1.0, 1.0, processed.shape) * scale
            
            processed += noise.astype(np.float32)
            
        return torch.from_numpy(processed)

    @staticmethod
    def normalize_peak(waveform: torch.Tensor, target_db: float = -1.0) -> torch.Tensor:
        """
        Normalizes waveform to a specific peak dB level.
        """
        peak = torch.max(torch.abs(waveform))
        
        if peak == 0:
            return waveform
            
        target_linear = 10.0 ** (target_db / 20.0)
        gain = target_linear / peak
        
        return waveform * gain

    @staticmethod
    def match_loudness(source: torch.Tensor, target: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Matches the loudness of the target waveform to the source waveform using LUFS.
        Returns the gain-adjusted target waveform.
        """
        meter = pyln.Meter(sr)
        
        # Convert to numpy (Samples, Channels) for pyloudnorm
        src_np = source.cpu().t().numpy()
        tgt_np = target.cpu().t().numpy()
        
        try:
            src_lufs = meter.integrated_loudness(src_np)
            tgt_lufs = meter.integrated_loudness(tgt_np)
            
            if np.isinf(src_lufs) or np.isinf(tgt_lufs):
                return target
                
            delta = src_lufs - tgt_lufs
            gain_linear = 10.0 ** (delta / 20.0)
            
            return target * gain_linear
            
        except ValueError:
            # Silence or error
            return target
