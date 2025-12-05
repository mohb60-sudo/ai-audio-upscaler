import unittest
import torch
import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.restoration import StemRestorer

class TestStemRestorer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.restorer = StemRestorer(self.device)
        self.sr = 44100
        
    def test_analyze_stem_structure(self):
        """Verifies analyze_stem returns correct keys."""
        waveform = torch.randn(2, 44100)
        profile = self.restorer.analyze_stem(waveform, self.sr, "vocals")
        
        self.assertIn("de_hiss", profile)
        self.assertIn("de_reverb", profile)
        self.assertIn("transient_expand", profile)
        self.assertIn("mono_bass", profile)

    def test_mono_maker_logic(self):
        """Verifies that Mono-Maker reduces stereo width for Bass using M/S."""
        # Create a wide stereo signal (Left and Right are different)
        t = torch.linspace(0, 1, 44100)
        left = torch.sin(2 * np.pi * 100 * t) # 100Hz
        right = torch.cos(2 * np.pi * 100 * t) # 100Hz (Phase shifted)
        wide_bass = torch.stack([left, right])
        
        # Analyze
        profile = self.restorer.analyze_stem(wide_bass, self.sr, "bass")
        self.assertEqual(profile["mono_bass"], 1.0)
        
        # Restore
        restored = self.restorer.restore_stem(wide_bass, self.sr, "bass", profile, strength=1.0)
        
        # Check if Side channel energy is reduced
        # Side = (L-R)/2
        input_side = (wide_bass[0] - wide_bass[1]) / 2
        output_side = (restored[0] - restored[1]) / 2
        
        input_side_energy = torch.mean(input_side**2)
        output_side_energy = torch.mean(output_side**2)
        
        # Output side energy should be significantly lower (since 100Hz < 200Hz cutoff)
        self.assertLess(output_side_energy, input_side_energy)

    def test_limiter_logic(self):
        """Verifies that the safety limiter prevents clipping."""
        # Create a massive signal
        waveform = torch.randn(2, 44100) * 10.0
        profile = {"transient_expand": 0.0, "de_hiss": 0.0, "mono_bass": 0.0, "de_reverb": 0.0}
        
        restored = self.restorer.restore_stem(waveform, self.sr, "drums", profile, strength=1.0)
        
        # Should be clamped by tanh (approx -1 to 1)
        self.assertLess(restored.max(), 1.05)
        self.assertGreater(restored.min(), -1.05)

    def test_transient_shaper_logic(self):
        """Verifies that Transient Shaper modifies the signal."""
        waveform = torch.randn(2, 44100)
        profile = {"transient_expand": 1.0, "de_hiss": 0.0, "mono_bass": 0.0, "de_reverb": 0.0}
        
        restored = self.restorer.restore_stem(waveform, self.sr, "drums", profile, strength=1.0)
        
        # Should be different (EQ applied)
        self.assertFalse(torch.allclose(waveform, restored))

if __name__ == "__main__":
    unittest.main()
