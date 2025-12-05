from .pre_processor import AudioPreProcessor
from .post_processor import AudioMasteringChain
import torch

class AudioChain:
    """
    Orchestrates the full audio processing pipeline:
    Input -> Pre-Processing -> AI (External) -> Post-Processing -> Output
    """
    
    def __init__(self, sr: int):
        self.sr = sr
        self.pre = AudioPreProcessor()
        self.post = AudioMasteringChain(sr)
        
    def pre_process(self, waveform: torch.Tensor, sr: int = None, remove_dc: bool = True, normalize_lufs: bool = True) -> tuple[torch.Tensor, float]:
        """
        Condition audio before AI.
        Returns (processed_waveform, gain_applied_db)
        """
        if sr is None:
            sr = self.sr
            
        out = waveform
        
        if remove_dc:
            out = self.pre.remove_dc_offset(out, sr)
            
        # De-click always on for safety? Or optional?
        # Let's make it optional but default off in UI for now to save CPU
        # out = self.pre.de_click(out)
            
        gain = 0.0
        if normalize_lufs:
            out, gain = self.pre.normalize_lufs(out, sr, target_lufs=-18.0)
            
        return out, gain
        
    def post_process(self, waveform: torch.Tensor, gain_compensation: float = 0.0, 
                     limit: bool = True, saturation: float = 0.0, 
                     stereo_width: float = 1.0, transient_shaping: float = 0.0,
                     dither: bool = False) -> torch.Tensor:
        """
        Mastering chain.
        """
        # Apply gain compensation if we normalized input
        # If we lowered input by 6dB, we might want to raise output by 6dB before limiting
        # But AI model might have changed loudness.
        # Generally, we rely on the Limiter to bring it up to target.
        
        # Apply mastering effects
        out = self.post.process(waveform, limit=limit, saturation=saturation, 
                                stereo_width=stereo_width, transient_shaping=transient_shaping,
                                dither=dither)
        
        return out
