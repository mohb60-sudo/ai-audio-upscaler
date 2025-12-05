import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.ai_upscaler.diffusion import DiffusionScheduler, DiffusionUNet
from ai_audio_upscaler.ai_upscaler.model import AudioSuperResNet
from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from ai_audio_upscaler.config import UpscalerConfig

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.channels = 1
        self.length = 16384
        self.time_emb_dim = 32
        
    def test_scheduler_noise(self):
        """Test noise addition and shape preservation."""
        scheduler = DiffusionScheduler(num_steps=100)
        x_start = torch.randn(self.batch_size, self.channels, self.length)
        t = torch.randint(0, 100, (self.batch_size,))
        
        noise = torch.randn_like(x_start)
        x_noisy, _ = scheduler.add_noise(x_start, t, noise)
        
        self.assertEqual(x_noisy.shape, x_start.shape)
        self.assertFalse(torch.allclose(x_noisy, x_start)) # Should be different
        
    def test_unet_forward(self):
        """Test DiffusionUNet forward pass with time embeddings."""
        # Base model must accept 2 channels (Input + Condition)
        base_model = AudioSuperResNet(in_channels=2, base_channels=16, num_layers=2, time_emb_dim=self.time_emb_dim)
        model = DiffusionUNet(base_model)
        
        x = torch.randn(self.batch_size, 1, self.length)
        t = torch.randint(0, 100, (self.batch_size,))
        condition = torch.randn(self.batch_size, 1, self.length) # Low-res condition
        
        output = model(x, t, condition=condition)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_inference_wrapper_integration(self):
        """Test AIUpscalerWrapper with Diffusion configuration."""
        # Create a dummy checkpoint with diffusion config
        config = UpscalerConfig(device='cpu', model_checkpoint="dummy_diffusion.ckpt")
        
        # Mock the checkpoint loading to avoid needing a real file
        # We'll manually initialize the wrapper's model instead
        wrapper = AIUpscalerWrapper(config)
        
        # Manually inject diffusion model
        # Base model must accept 2 channels
        base_model = AudioSuperResNet(in_channels=2, base_channels=16, num_layers=2, time_emb_dim=32)
        wrapper.model = DiffusionUNet(base_model)
        wrapper.diffusion_scheduler = DiffusionScheduler(num_steps=50) # Small steps for test
        wrapper.use_diffusion = True
        wrapper.device = self.device
        wrapper.model.to(self.device)
        
        # Run enhance
        waveform = torch.randn(1, self.length) # (Channels, Time)
        output = wrapper.enhance(waveform, original_sr=24000, target_sr=48000, diffusion_steps=10)
        
        self.assertEqual(output.shape, waveform.shape)
        self.assertFalse(torch.isnan(output).any())

if __name__ == '__main__':
    unittest.main()
