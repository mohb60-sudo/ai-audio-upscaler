from dataclasses import dataclass
from typing import Optional

@dataclass
class UpscalerConfig:
    """
    Configuration for the Audio Upscaler Pipeline.
    
    Attributes:
        target_sample_rate (int): Target sample rate in Hz (e.g., 48000, 96000).
        mode (str): Upscaling mode. 'baseline' (DSP only) or 'ai' (Neural Network).
        baseline_method (str): DSP interpolation method (e.g., 'poly-sinc-hq').
        model_checkpoint (Optional[str]): Path to the AI model checkpoint file.
        device (str): Compute device ('cpu' or 'cuda').
        export_format (str): Output file format ('wav', 'flac', etc.).
        
        # Advanced DSP Settings
        use_advanced_dsp (bool): Enable HQPlayer-inspired DSP chain.
        dsp_quality_preset (str): DSP quality level ('fast', 'balanced', 'quality', 'ultra').
        apply_dither (bool): Apply TPDF dither to the final output.
        noise_shaper (str): Noise shaping algorithm ('auto', 'tpdf', 'ns9').
        
        # AI / Diffusion Settings
        use_diffusion (bool): Enable Diffusion-based upscaling (vs GAN).
        diffusion_steps (int): Number of denoising steps for Diffusion (e.g., 50).
        model_type (str): Architecture type ('gan' or 'diffusion').
    """
    target_sample_rate: int = 48000
    mode: str = "baseline"
    baseline_method: str = "sinc"
    model_checkpoint: Optional[str] = None
    device: str = "cpu"
    export_format: str = "wav"
    
    # Advanced DSP settings (HQPlayer-inspired)
    use_advanced_dsp: bool = False
    dsp_quality_preset: str = "balanced"
    apply_dither: bool = True
    noise_shaper: str = "auto"

    # Diffusion Settings
    use_diffusion: bool = False
    diffusion_steps: int = 1000
    model_type: str = "gan"
