
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

class MockDiscriminator:
    def score_candidate(self, candidate, chunk_seconds=None, sr=None):
        # Always return 0.0 to force rejection
        return 0.0

def test_qc_fallback():
    logger.info("Testing QC Fallback Logic...")
    config = UpscalerConfig()
    config.device = 'cpu'
    
    wrapper = AIUpscalerWrapper(config)
    
    # Inject Mock Discriminator
    wrapper.discriminator = MockDiscriminator()
    
    # Mock enhance to return random tensors
    def mock_enhance(*args, **kwargs):
        return torch.randn(1, 48000) # 1 second @ 48k

    wrapper.enhance = mock_enhance
    
    # Run QC with high threshold
    # Judge returns 0.0, Threshold is 0.5 -> All should be rejected
    output = wrapper.enhance_with_qc(
        torch.randn(1, 48000), 
        original_sr=48000, 
        target_sr=48000, 
        chunk_seconds=0.5, 
        candidate_count=4, 
        judge_threshold=0.5
    )
    
    logger.info(f"Output Shape: {output.shape}")
    assert output is not None
    logger.info("✅ QC Fallback Test Passed (Output produced despite Judge rejection)")

if __name__ == "__main__":
    test_qc_fallback()
