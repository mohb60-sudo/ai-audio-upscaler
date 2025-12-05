import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import unittest
import numpy as np
from ai_audio_upscaler.dsp.post_processor import AudioMasteringChain
import pyloudnorm as pyln

class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        self.post = AudioMasteringChain(self.sr)
        
    def test_normalize_peak(self):
        # Create a sine wave with peak 0.5
        t = torch.linspace(0, 1, self.sr)
        waveform = 0.5 * torch.sin(2 * np.pi * 440 * t).unsqueeze(0) # (1, T)
        
        # Normalize to -1dB (approx 0.891)
        target_db = -1.0
        normalized = self.post.normalize_peak(waveform, target_db=target_db)
        
        peak = torch.max(torch.abs(normalized))
        target_linear = 10.0 ** (target_db / 20.0)
        
        print(f"Original Peak: {torch.max(torch.abs(waveform)):.4f}")
        print(f"Target Peak: {target_linear:.4f} (-1dB)")
        print(f"Actual Peak: {peak:.4f}")
        
        self.assertTrue(torch.isclose(peak, torch.tensor(target_linear), atol=1e-4))
        
    def test_match_loudness(self):
        # Create source: Loud sine wave (-6dB)
        t = torch.linspace(0, 1, self.sr)
        src_wave = 0.5 * torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        # Create target: Quiet sine wave (-20dB)
        tgt_wave = 0.1 * torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        # Measure initial LUFS
        meter = pyln.Meter(self.sr)
        src_lufs = meter.integrated_loudness(src_wave.cpu().t().numpy())
        tgt_lufs_initial = meter.integrated_loudness(tgt_wave.cpu().t().numpy())
        
        print(f"Source LUFS: {src_lufs:.2f}")
        print(f"Target LUFS (Initial): {tgt_lufs_initial:.2f}")
        
        # Match Loudness
        matched_wave = self.post.match_loudness(src_wave, tgt_wave, self.sr)
        
        # Measure result
        tgt_lufs_final = meter.integrated_loudness(matched_wave.cpu().t().numpy())
        print(f"Target LUFS (Final): {tgt_lufs_final:.2f}")
        
        self.assertTrue(np.isclose(src_lufs, tgt_lufs_final, atol=0.1))

if __name__ == '__main__':
    unittest.main()
