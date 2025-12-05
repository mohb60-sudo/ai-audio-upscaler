import unittest
import os
import shutil
import tempfile
import torch
import torchaudio
from unittest.mock import MagicMock, patch
from web_app.app import process_batch

class TestBatchProcessing(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for inputs and outputs
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create dummy audio files
        self.sr = 44100
        self.audio = torch.randn(2, self.sr * 1) # 1 second stereo
        
        self.file1 = os.path.join(self.input_dir, "test1.wav")
        self.file2 = os.path.join(self.input_dir, "test2.wav")
        
        torchaudio.save(self.file1, self.audio, self.sr)
        torchaudio.save(self.file2, self.audio, self.sr)
        
        self.files = [self.file1, self.file2]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_process_batch_baseline(self):
        """Test batch processing in Baseline mode (fast)."""
        
        # Mock Progress
        mock_progress = MagicMock()
        
        # Call process_batch
        log = process_batch(
            files=self.files,
            target_rate=48000,
            mode="Baseline",
            baseline_method="Sinc",
            model_name=None,
            export_format="WAV",
            output_dir=self.output_dir,
            device="cpu",
            normalization_mode="Peak -1dB",
            dsp_quality="Fast",
            apply_dither=False,
            tta=False,
            stereo_mode="lr",
            transient_strength=0.0,
            spectral_matching=False,
            remove_dc=True,
            normalize_input=True,
            limit=True,
            saturation=0.0,
            stereo_width=1.0,
            transient_shaping=0.0,
            qc=False,
            candidate_count=2,
            judge_threshold=0.5,
            diffusion_steps=10,
            denoising_strength=0.6,
            progress=mock_progress
        )
        
        print(log)
        
        # Verify Output Files
        out1 = os.path.join(self.output_dir, "test1_upscaled.wav")
        out2 = os.path.join(self.output_dir, "test2_upscaled.wav")
        
        self.assertTrue(os.path.exists(out1), "Output file 1 not created")
        self.assertTrue(os.path.exists(out2), "Output file 2 not created")
        
        # Verify Log
        self.assertIn("Batch Job Complete", log)
        self.assertIn("test1.wav", log)
        self.assertIn("test2.wav", log)
        self.assertIn("✅ [OK]", log)
        
        # Verify Progress Calls
        # Should be called at least twice (once per file)
        self.assertTrue(mock_progress.call_count >= 2)

if __name__ == '__main__':
    unittest.main()
