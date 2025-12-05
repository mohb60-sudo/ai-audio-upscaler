import torch
import unittest
from unittest.mock import MagicMock, patch
from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from ai_audio_upscaler.config import UpscalerConfig

class TestPartialDiffusion(unittest.TestCase):
    def setUp(self):
        self.config = UpscalerConfig(
            target_sample_rate=48000,
            mode="ai",
            device="cpu",
            model_checkpoint=None
        )
        
        with patch("ai_audio_upscaler.ai_upscaler.inference.AudioSuperResNet"), \
             patch("ai_audio_upscaler.ai_upscaler.inference.DiffusionUNet"), \
             patch("ai_audio_upscaler.ai_upscaler.inference.DiffusionScheduler"):
            self.wrapper = AIUpscalerWrapper(self.config)
            
        self.wrapper.use_diffusion = True
        self.wrapper.model = MagicMock()
        self.wrapper.model.side_effect = lambda x, *args, **kwargs: torch.randn_like(x)
        
        self.wrapper.diffusion_scheduler = MagicMock()
        self.wrapper.diffusion_scheduler.num_steps = 50
        self.wrapper.device = torch.device("cpu")
        
        # Mock add_noise to return (noisy_x, noise)
        def mock_add_noise(x, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x)
            # Return x + noise (simplified)
            return x + noise, noise
        self.wrapper.diffusion_scheduler.add_noise.side_effect = mock_add_noise
        
        # Mock step/step_ddim
        self.wrapper.diffusion_scheduler.step.side_effect = lambda pred, t, x: x
        self.wrapper.diffusion_scheduler.step_ddim.side_effect = lambda pred, t, x, t_prev, eta=0.0, **kwargs: x

    def test_partial_diffusion_init(self):
        """Verify that x is initialized with add_noise at the correct timestep"""
        sr = 48000
        waveform = torch.zeros(1, sr) # Silence input
        
        # Mock _enhance_batch to spy on logic? No, we want to test _enhance_batch logic.
        # We need to call _enhance_batch directly.
        
        # Input: (Batch, Channels, Time)
        batch_tensor = torch.zeros(1, 1, sr)
        
        # Run _enhance_batch with diffusion_steps=25 (step_ratio=2)
        # timesteps should be [0, 2, ..., 48]
        # start_t should be 48
        
        self.wrapper._enhance_batch(
            batch_tensor, 
            diffusion_steps=25,
            denoising_strength=1.0
        )
        
        # Verify add_noise was called
        self.assertTrue(self.wrapper.diffusion_scheduler.add_noise.called)
        
        # Check arguments
        args, _ = self.wrapper.diffusion_scheduler.add_noise.call_args
        x_start, t, noise = args
        
        print(f"add_noise called with t={t}")
        
        # t should be a tensor of [48]
        self.assertEqual(t.item(), 48)
        
        # Verify x_start is the input batch
        self.assertTrue(torch.equal(x_start, batch_tensor))
        
        print("Verified: Partial diffusion initialized at t=48 for 25 steps (Ratio=2)")

if __name__ == '__main__':
    unittest.main()
