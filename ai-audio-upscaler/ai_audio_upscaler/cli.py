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
    
    parser.add_argument("input_path", type=str, help="Path to input audio file (.wav)")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save output file")
    parser.add_argument("--target-rate", type=int, default=48000, help="Target sample rate (Hz)")
    parser.add_argument("--mode", type=str, choices=["baseline", "ai"], default="baseline", help="Upscaling mode")
    parser.add_argument("--baseline-method", type=str, choices=["sinc", "linear"], default="sinc", help="Baseline resampling method")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to AI model checkpoint (.pt/.ckpt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")

    args = parser.parse_args()

    # Determine output path if not provided
    if not args.output_path:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_upscaled_{args.target_rate}hz{ext}"

    config = UpscalerConfig(
        target_sample_rate=args.target_rate,
        mode=args.mode,
        baseline_method=args.baseline_method,
        model_checkpoint=args.model_checkpoint,
        device=args.device
    )

    try:
        pipeline = AudioUpscalerPipeline(config)
        pipeline.run(args.input_path, args.output_path)
        logger.info(f"Success! Output saved to {args.output_path}")
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
