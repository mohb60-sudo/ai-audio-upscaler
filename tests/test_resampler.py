import unittest
import torch
import os
import shutil
import torchaudio
from ai_audio_upscaler.dsp_basic import DSPUpscaler
from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline

class TestUpscaler(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a dummy input file
        self.sr = 16000
        self.input_path = os.path.join(self.test_dir, "input.wav")
        waveform = torch.randn(1, 16000) # 1 second of noise
        torchaudio.save(self.input_path, waveform, self.sr)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dsp_sinc(self):
        target_sr = 24000
        dsp = DSPUpscaler(target_sr, method="sinc")
        waveform, _ = torchaudio.load(self.input_path)
        
        out = dsp.process(waveform, self.sr)
        
        self.assertEqual(out.shape[1], 24000)
        self.assertEqual(out.shape[0], 1)

    def test_pipeline_baseline(self):
        target_sr = 32000
        output_path = os.path.join(self.test_dir, "output_baseline.wav")
        
        config = UpscalerConfig(target_sample_rate=target_sr, mode="baseline")
        pipeline = AudioUpscalerPipeline(config)
        pipeline.run(self.input_path, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        w, sr = torchaudio.load(output_path)
        self.assertEqual(sr, target_sr)

    def test_pipeline_ai_random(self):
        # Test AI mode with random weights (no crash)
        target_sr = 32000
        output_path = os.path.join(self.test_dir, "output_ai.wav")
        
        config = UpscalerConfig(target_sample_rate=target_sr, mode="ai")
        pipeline = AudioUpscalerPipeline(config)
        pipeline.run(self.input_path, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        w, sr = torchaudio.load(output_path)
        self.assertEqual(sr, target_sr)

if __name__ == '__main__':
    unittest.main()
