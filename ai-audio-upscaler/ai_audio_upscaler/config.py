from dataclasses import dataclass
from typing import Optional

@dataclass
class UpscalerConfig:
    """Configuration for the Audio Upscaler."""
    target_sample_rate: int = 48000
    mode: str = "baseline"  # 'baseline' or 'ai'
    baseline_method: str = "sinc"  # 'sinc' or 'linear'
    model_checkpoint: Optional[str] = None
    device: str = "cpu"  # 'cpu' or 'cuda'
    export_format: str = "wav"  # 'wav', 'flac', 'mp3', 'ogg'
