import argparse
import logging
import sys
import os

# Add project root to path so we can run this file directly if needed, 
# though running via `python -m ai_audio_upscaler` is preferred.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    logger = logging.getLogger("CLI")

    parser = argparse.ArgumentParser(description="AI Audio Upscaler CLI")
    
    # Core Settings
    parser.add_argument("input_path", type=str, help="Path to input audio file (.wav)")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save output file")
    parser.add_argument("--target-rate", type=int, default=48000, help="Target sample rate (Hz)")
    parser.add_argument("--mode", type=str, choices=["baseline", "ai"], default="baseline", help="Upscaling mode")
    parser.add_argument("--baseline-method", type=str, choices=["sinc", "linear", "poly-sinc", "poly-sinc-hq", "poly-sinc-fast"], default="sinc", help="Resampling method")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to AI model checkpoint (.pt/.ckpt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    
    # Advanced DSP Settings
    parser.add_argument("--dsp-quality", type=str, choices=["fast", "balanced", "quality", "ultra"], default="balanced", help="DSP quality preset")
    parser.add_argument("--no-dither", action="store_false", dest="apply_dither", help="Disable dithering")
    parser.set_defaults(apply_dither=True)

    # AI Post-Processing
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (slower, better quality)")
    parser.add_argument("--stereo-mode", type=str, choices=["lr", "ms"], default="lr", help="Stereo processing mode (Left/Right or Mid/Side)")
    parser.add_argument("--transient-strength", type=float, default=0.0, help="Transient restoration strength (0.0-1.0)")
    parser.add_argument("--spectral-matching", action="store_true", help="Enable spectral matching")

    # Mastering Chain
    parser.add_argument("--no-limit", action="store_false", dest="limit", help="Disable output limiting")
    parser.add_argument("--norm-mode", type=str, default="Peak -1dB", choices=["Match Source", "Peak -1dB", "Peak -3dB", "None"], help="Normalization mode")
    parser.add_argument("--remove-dc", action="store_true", help="Remove DC offset from input")
    parser.add_argument("--normalize-input", action="store_true", help="Normalize input to -18 LUFS before processing")
    parser.add_argument("--saturation", type=float, default=0.0, help="Tube saturation amount (0.0-1.0)")
    parser.add_argument("--stereo-width", type=float, default=1.0, help="Stereo width (0.0=Mono, 1.0=Normal, >1.0=Wide)")
    parser.add_argument("--transient-shaping", type=float, default=0.0, help="Transient shaping amount (0.0-1.0)")

    # Quality Control
    parser.add_argument("--qc", action="store_true", help="Enable Hybrid Quality Control (Judge + Consensus)")
    parser.add_argument("--qc-candidates", type=int, default=8, help="Number of candidates to generate for QC")
    parser.add_argument("--qc-threshold", type=float, default=0.5, help="Discriminator threshold for QC (0.0-1.0)")

    parser.set_defaults(limit=True)

    args = parser.parse_args()

    # Determine output path if not provided
    if not args.output_path:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_upscaled_{args.target_rate}hz{ext}"

    # Map baseline method to config
    use_advanced_dsp = args.baseline_method.startswith("poly-sinc")

    config = UpscalerConfig(
        target_sample_rate=args.target_rate,
        mode=args.mode,
        baseline_method=args.baseline_method,
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        use_advanced_dsp=use_advanced_dsp,
        dsp_quality_preset=args.dsp_quality,
        apply_dither=args.apply_dither
    )

    try:
        pipeline = AudioUpscalerPipeline(config)
        
        # Run pipeline with all arguments
        pipeline.run(
            args.input_path, 
            args.output_path,
            normalization_mode=args.norm_mode,
            generate_analysis=False, # CLI doesn't need plots
            tta=args.tta,
            stereo_mode=args.stereo_mode,
            transient_strength=args.transient_strength,
            spectral_matching=args.spectral_matching,
            remove_dc=args.remove_dc,
            normalize_input=args.normalize_input,
            limit=args.limit,
            saturation=args.saturation,
            stereo_width=args.stereo_width,
            transient_shaping=args.transient_shaping,
            qc=args.qc,
            candidate_count=args.qc_candidates,
            judge_threshold=args.qc_threshold
        )
        logger.info(f"Success! Output saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
