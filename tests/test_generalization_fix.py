import unittest
import torch
import torchaudio
import os
import sys
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.ai_upscaler.transforms import BandwidthLimiter
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
from ai_audio_upscaler.config import UpscalerConfig

class TestGeneralizationFix(unittest.TestCase):
    def test_bandwidth_limiter_range(self):
        """Verify BandwidthLimiter can produce cutoffs > 16kHz."""
        # Setup: Target SR 96k, Max Cutoff 24k (Nyquist of 48k input)
        limiter = BandwidthLimiter(sample_rate=96000, max_cutoff=24000)
        
        # We can't easily check the internal random choice without mocking,
        # but we can check if the object was initialized correctly.
        self.assertEqual(limiter.max_cutoff, 24000)
        
        # Run it a few times to ensure no crash
        wave = torch.randn(1, 96000) # 1 sec
        out = limiter(wave)
        self.assertEqual(out.shape, wave.shape)

    def test_24bit_export(self):
        """Verify pipeline exports 24-bit FLAC/WAV."""
        config = UpscalerConfig()
        config.export_format = "flac"
        pipeline = AudioUpscalerPipeline(config)
        
        wave = torch.randn(1, 48000)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_24bit.flac")
            pipeline.save_audio(wave, out_path, 48000)
            
            # Verify file exists
            self.assertTrue(os.path.exists(out_path))
            
            # Verify metadata (torchaudio.info)
            info = torchaudio.info(out_path)
            # FLAC bit depth is usually in bits_per_sample
            self.assertEqual(info.bits_per_sample, 24)
            
            # Test WAV 24-bit
            config.export_format = "wav"
            out_path_wav = os.path.join(tmpdir, "test_24bit.wav")
            pipeline.save_audio(wave, out_path_wav, 48000)
            
            info_wav = torchaudio.info(out_path_wav)
            # WAV 24-bit might be reported as 24 or 32 depending on backend, 
            # but let's check what torchaudio reports.
            # Note: 24-bit PCM in WAV is standard.
            self.assertEqual(info_wav.bits_per_sample, 24)

if __name__ == '__main__':
    unittest.main()
