
import torch
import logging
import os
import sys
import shutil
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from ai_audio_upscaler.config import UpscalerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vram_estimation():
    logger.info("Testing VRAM Estimation...")
    config = UpscalerConfig()
    # Force CPU for test environment (unless CUDA is available)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wrapper = AIUpscalerWrapper(config)
    
    # Mock model attributes if they don't exist (in case of fallback)
    if not hasattr(wrapper.model, 'base_channels'):
        wrapper.model.base_channels = 32
        wrapper.model.num_layers = 4
        
    # Test estimation
    chunk_size = 48000 * 2 # 2 seconds at 48k
    batch_size = wrapper.estimate_max_batch_size(chunk_size)
    logger.info(f"Estimated Batch Size: {batch_size}")
    
    assert batch_size >= 1, "Batch size should be at least 1"
    logger.info("✅ VRAM Estimation Test Passed")

def test_qc_streaming():
    logger.info("Testing QC Streaming Logic...")
    config = UpscalerConfig()
    config.device = 'cpu' # Force CPU for logic test
    
    wrapper = AIUpscalerWrapper(config)
    
    # Mock enhance method to avoid running actual heavy inference
    # We just want to test the QC flow (save to disk, load, consensus)
    def mock_enhance(*args, **kwargs):
        # Return random tensor of correct shape
        # waveform is 1st arg
        waveform = args[0]
        target_sr = args[2]
        chunk_seconds = args[3]
        
        # Calculate expected output shape
        # Upsampling is handled by DSP before enhance usually, but here enhance takes (C, T)
        # and returns (C, T).
        # In the real pipeline, enhance receives upsampled audio.
        return torch.randn_like(waveform)

    # Monkey patch enhance
    wrapper.enhance = mock_enhance
    
    # Create dummy input
    sr = 48000
    duration = 1.0
    waveform = torch.randn(1, int(sr * duration))
    
    # Run QC
    # We expect it to generate candidates, save them, run consensus, and return mean
    output = wrapper.enhance_with_qc(
        waveform, 
        original_sr=sr, 
        target_sr=sr, 
        chunk_seconds=0.5, 
        candidate_count=3, 
        judge_threshold=0.0 # Pass everything
    )
    
    logger.info(f"Output Shape: {output.shape}")
    assert output.shape == waveform.shape, "Output shape mismatch"
    logger.info("✅ QC Streaming Test Passed")

if __name__ == "__main__":
    try:
        test_vram_estimation()
        test_qc_streaming()
        logger.info("🎉 ALL TESTS PASSED")
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
