import torch
import logging
import os
from ..config import UpscalerConfig
from .model import AudioSuperResNet

logger = logging.getLogger(__name__)

class AIUpscalerWrapper:
    """
    Wraps the neural model for inference.
    """
    def __init__(self, config: UpscalerConfig):
        self.config = config
        
        # Determine device
        if config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if config.device == 'cuda':
                logger.warning("CUDA requested but not available. Falling back to CPU.")
        
        logger.info(f"AI Model initialized on device: {self.device}")
        
        # Initialize model
        # Note: We assume mono or stereo processing. 
        # Ideally, we process channels independently or use a model that handles C channels.
        # For simplicity, we'll process (Batch, Channels, Time) where Channels is the audio channels.
        # But our model defaults to in_channels=1. If we have stereo, we can treat it as batch=2 or change model.
        # Let's stick to treating channels as independent batch items for the 1D model (Spatial/Channel independence).
        self.model = AudioSuperResNet(in_channels=1, hidden_channels=32, num_blocks=4)
        self.model.to(self.device)
        self.model.eval()

        if config.model_checkpoint and os.path.exists(config.model_checkpoint):
            logger.info(f"Loading model checkpoint: {config.model_checkpoint}")
            try:
                state_dict = torch.load(config.model_checkpoint, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        else:
            logger.warning("No valid checkpoint provided. Using RANDOM INITIALIZED weights. Output will be noisy/garbage!")

    @staticmethod
    def list_available_models(models_dir="checkpoints"):
        """Scans the checkpoints directory for .pth or .ckpt files."""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        return [f for f in os.listdir(models_dir) if f.endswith(".pth") or f.endswith(".ckpt")]

    def enhance(self, waveform: torch.Tensor, original_sr: int, target_sr: int, chunk_seconds: float = 2.0) -> torch.Tensor:
        """
        Runs the neural network on the waveform using chunking to avoid OOM.
        
        Args:
            waveform: (Channels, Time) tensor.
            chunk_seconds: Length of each chunk in seconds.
        """
        channels, time = waveform.shape
        chunk_size = int(target_sr * chunk_seconds)
        
        # If short enough, run directly
        if time <= chunk_size:
            return self._enhance_batch(waveform)
        
        logger.info(f"Audio length {time} samples > chunk size {chunk_size}. Using chunked inference.")
        
        output_chunks = []
        
        # Process in chunks
        # TODO: Implement overlap-add for better quality at boundaries
        for i in range(0, time, chunk_size):
            chunk = waveform[:, i:i + chunk_size]
            
            # Pad last chunk if needed (though model handles variable length, 
            # consistent batch size might be better, but here we just pass it)
            
            processed_chunk = self._enhance_batch(chunk)
            output_chunks.append(processed_chunk)
            
            # Optional: Clear cache if really tight on memory
            # torch.cuda.empty_cache()
            
        return torch.cat(output_chunks, dim=1)

    def _enhance_batch(self, waveform_chunk: torch.Tensor) -> torch.Tensor:
        """Helper to run inference on a single chunk."""
        with torch.no_grad():
            # Prepare input: (Channels, Time) -> (Batch=Channels, 1, Time)
            x = waveform_chunk.unsqueeze(1).to(self.device)
            
            try:
                out = self.model(x)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("OOM Error inside chunk. Try reducing chunk_seconds.")
                    raise e
                raise e
            
            # Reshape back: (Channels, 1, Time) -> (Channels, Time)
            out = out.squeeze(1).cpu()
            
            # Clamp to valid audio range [-1, 1]
            out = torch.clamp(out, -1.0, 1.0)
            
            return out
