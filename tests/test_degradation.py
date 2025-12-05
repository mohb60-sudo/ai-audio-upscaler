import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.ai_upscaler.degradation import AdvancedDegradation

class TestDegradation(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.deg = AdvancedDegradation(sample_rate=self.sr)
        self.audio = torch.randn(1, 1, self.sr * 2) # 2 seconds of stereo audio (but shape is 1, 1, L for mono test)
        # Actually AdvancedDegradation expects (Channels, Time) or (Batch, Channels, Time)
        # Let's use (1, 48000)
        self.audio = torch.randn(1, 48000)

    def test_output_shape(self):
        """Test that degradation preserves shape."""
        out = self.deg(self.audio)
        self.assertEqual(out.shape, self.audio.shape)
        
    def test_clipping(self):
        """Test that clipping actually clips."""
        # Create high amplitude signal
        audio = torch.randn(1, 1, 48000) * 10.0
        # Call method directly
        out = self.deg.apply_clipping(audio)
        
        self.assertFalse(torch.allclose(out, audio))
        # Check if values are reduced (soft clipping)
        self.assertTrue(out.abs().max() < audio.abs().max())
        
    def test_bandwidth_limiting(self):
        """Test bandwidth limiting (resampling)."""
        # Mock random.randint to return a low cutoff (4000) to ensure effect
        with unittest.mock.patch('random.randint', return_value=4000):
            # Call method directly
            out = self.deg.apply_bandwidth_limit(self.audio.unsqueeze(0))
            self.assertEqual(out.shape, (1, 1, 48000))
            self.assertFalse(torch.allclose(out, self.audio.unsqueeze(0)))

if __name__ == '__main__':
    unittest.main()
