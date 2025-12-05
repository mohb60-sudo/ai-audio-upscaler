import unittest
import torch
import numpy as np
from ai_audio_upscaler.analysis import detect_cutoff_frequency, analyze_file_quality
from unittest.mock import patch

class TestForensics(unittest.TestCase):
    def setUp(self):
        self.sr = 96000
        self.duration = 1.0
        self.t = torch.linspace(0, self.duration, int(self.sr * self.duration))
        
    def create_signal(self, max_freq, noise_level_db=-100):
        """Creates a signal with content up to max_freq."""
        # White noise base
        signal = torch.randn_like(self.t) * (10**(noise_level_db/20))
        
        # chirp: sin(2*pi * (f0 + (f1-f0)*t/T/2) * t)
        k = max_freq / self.duration
        chirp = torch.sin(2 * np.pi * (k/2) * self.t**2)
        
        # Apply window to reduce spectral leakage
        window = torch.hann_window(len(self.t))
        signal += chirp * window * 0.5
            
        return signal.unsqueeze(0) # (1, T)

    def test_true_high_res(self):
        """Test a file with content up to 30kHz."""
        # Signal with content up to 30kHz
        waveform = self.create_signal(30000)
        
        cutoff = detect_cutoff_frequency(waveform, self.sr, sensitivity="adaptive")
        print(f"True High-Res Cutoff: {cutoff} Hz")
        
        # Should be > 28kHz (allowing for some rolloff detection variance)
        self.assertGreater(cutoff, 28000)
        
        # Mocking load_audio_robust to return our synthetic waveform
        with patch('ai_audio_upscaler.analysis.load_audio_robust', return_value=(waveform, self.sr)):
            with patch('ai_audio_upscaler.analysis.get_audio_metadata', return_value={}):
                res = analyze_file_quality("dummy.wav")
                self.assertEqual(res["Status"], "✅ PASS")

    def test_fake_high_res(self):
        """Test a file with content only up to 15kHz (upsampled)."""
        waveform = self.create_signal(15000)
        
        cutoff = detect_cutoff_frequency(waveform, self.sr, sensitivity="adaptive")
        print(f"Fake High-Res Cutoff: {cutoff} Hz")
        
        # Should be around 15kHz
        self.assertLess(cutoff, 16000)
        
        with patch('ai_audio_upscaler.analysis.load_audio_robust', return_value=(waveform, self.sr)):
            with patch('ai_audio_upscaler.analysis.get_audio_metadata', return_value={}):
                res = analyze_file_quality("dummy.wav")
                self.assertEqual(res["Status"], "❌ FAIL")

    def test_borderline_high_res(self):
        """Test a file with content up to 20kHz (CD quality upsample)."""
        waveform = self.create_signal(20000)
        
        cutoff = detect_cutoff_frequency(waveform, self.sr, sensitivity="adaptive")
        print(f"Borderline Cutoff: {cutoff} Hz")
        
        with patch('ai_audio_upscaler.analysis.load_audio_robust', return_value=(waveform, self.sr)):
            with patch('ai_audio_upscaler.analysis.get_audio_metadata', return_value={}):
                res = analyze_file_quality("dummy.wav")
                # Should be WARN now (was FAIL before)
                self.assertEqual(res["Status"], "⚠️ WARN")

    def test_quiet_high_frequency(self):
        """Test high frequency content that is quiet (-70dB)."""
        # Base noise floor -100dB
        waveform = torch.randn_like(self.t) * (10**(-100/20))
        
        # Add loud tone at 1kHz (0dB reference)
        waveform += torch.sin(2 * np.pi * 1000 * self.t)
        
        # Add quiet tone at 25kHz (-70dB)
        # This would fail the old -60dB fixed threshold
        waveform += torch.sin(2 * np.pi * 25000 * self.t) * (10**(-70/20))
        
        waveform = waveform.unsqueeze(0)
        
        cutoff = detect_cutoff_frequency(waveform, self.sr, sensitivity="adaptive")
        print(f"Quiet HF Cutoff: {cutoff} Hz")
        
        # Should detect the 25kHz tone
        self.assertGreater(cutoff, 24000)

if __name__ == '__main__':
    unittest.main()
