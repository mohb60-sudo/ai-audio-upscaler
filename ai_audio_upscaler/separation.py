import torch
import torchaudio
import logging
import gc
from typing import Dict, Optional

# Try to import demucs, but don't crash if not installed (graceful degradation)
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SourceSeparator:
    """
    Wrapper around Demucs for source separation.
    Handles model loading, inference, and VRAM management.
    """
    def __init__(self, device: torch.device, model_name: str = "htdemucs"):
        self.device = device
        self.model_name = model_name
        self.model = None
        
        if not DEMUCS_AVAILABLE:
            logger.warning("Demucs not installed. Source separation will be unavailable.")

    def load_model(self):
        """Loads the Demucs model into memory."""
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs is not installed. Please install it via 'pip install demucs'.")
            
        if self.model is not None:
            return

        logger.info(f"Loading Demucs model: {self.model_name}...")
        try:
            # Get the model (downloads if needed)
            self.model = pretrained.get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Demucs model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise

    def unload_model(self):
        """Unloads the model to free VRAM."""
        if self.model is not None:
            logger.info("Unloading Demucs model...")
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    def separate(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
        """
        Separates the input waveform into stems.
        
        Args:
            waveform: (Channels, Time) tensor.
            sample_rate: Sample rate of the input.
            
        Returns:
            Dict[str, Tensor]: Dictionary of stems (e.g., {'drums': ..., 'bass': ...}).
        """
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs is not installed.")
            
        if self.model is None:
            self.load_model()

        # Demucs expects (Batch, Channels, Time)
        if waveform.dim() == 2:
            x = waveform.unsqueeze(0)
        else:
            x = waveform

        # Demucs expects 44.1kHz. It handles resampling internally in `apply_model` if configured,
        # but explicit resampling gives us more control. However, `apply_model` is robust.
        # We will let `apply_model` handle the shift if possible, or we can just pass it.
        # Actually, `apply_model` takes `shifts` and `split` args, but assumes input matches model SR?
        # Let's check Demucs docs or source. Usually `apply_model` does NOT resample.
        # We should resample to model.samplerate (usually 44100).
        
        model_sr = self.model.samplerate
        if sample_rate != model_sr:
            logger.info(f"Resampling for Demucs: {sample_rate} -> {model_sr}")
            x = torchaudio.functional.resample(x, sample_rate, model_sr)

        logger.info("Running Demucs separation...")
        ref = x.mean(0)
        x = (x - ref.mean()) / ref.std() # Normalize for Demucs? Demucs usually handles its own norm.
        # Actually, Demucs `apply_model` expects raw audio. Let's stick to standard usage.
        # Revert manual norm.
        
        # Reload x to be safe
        if waveform.dim() == 2:
            x = waveform.unsqueeze(0)
        else:
            x = waveform
            
        if sample_rate != model_sr:
            x = torchaudio.functional.resample(x, sample_rate, model_sr)

        # Normalize: Demucs works best with standard levels.
        # We'll trust the input is reasonable (e.g. -1 to 1).
        
        with torch.no_grad():
            # apply_model(model, mix, shifts=1, split=True, overlap=0.25, transition_power=1., progress=False, device=None, num_workers=None, segment=None)
            # shifts=1 means 1 random shift (default). 0 is faster? 
            # shifts > 0 improves quality (shift invariance).
            sources = apply_model(self.model, x, shifts=0, split=True, overlap=0.25, device=self.device)
            
        # sources shape: (Batch, Sources, Channels, Time)
        # Sources are usually: drums, bass, other, vocals (in that order for htdemucs)
        source_names = self.model.sources
        
        stems = {}
        for i, name in enumerate(source_names):
            stem = sources[0, i] # (Channels, Time)
            
            # Resample back if needed
            if sample_rate != model_sr:
                stem = torchaudio.functional.resample(stem, model_sr, sample_rate)
                
            # Ensure length matches original (resampling might cause slight rounding errors)
            if stem.shape[-1] != waveform.shape[-1]:
                if stem.shape[-1] > waveform.shape[-1]:
                    stem = stem[..., :waveform.shape[-1]]
                else:
                    stem = torch.nn.functional.pad(stem, (0, waveform.shape[-1] - stem.shape[-1]))
            
            stems[name] = stem
            
        return stems
