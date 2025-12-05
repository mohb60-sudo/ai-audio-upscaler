import unittest
import torch
from unittest.mock import MagicMock, patch
from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from ai_audio_upscaler.config import UpscalerConfig

class TestDDIMSampling(unittest.TestCase):
    def setUp(self):
        self.config = UpscalerConfig(
            target_sample_rate=48000,
            mode="ai",
            device="cpu",
            model_checkpoint="dummy_ckpt.pth"
        )
        
        # Mock the model loading to avoid needing real weights
        with patch('ai_audio_upscaler.ai_upscaler.inference.torch.load') as mock_load, \
             patch('ai_audio_upscaler.ai_upscaler.inference.os.path.exists') as mock_exists:
            mock_exists.return_value = False # Force random init
            self.wrapper = AIUpscalerWrapper(self.config)
            
        # Force diffusion mode
        self.wrapper.use_diffusion = True
        self.wrapper.diffusion_scheduler = MagicMock()
        self.wrapper.diffusion_scheduler.num_steps = 1000
        # Ensure step and step_ddim return a tensor of correct shape (1, 1, 48000)
        self.wrapper.diffusion_scheduler.step.return_value = torch.zeros(1, 1, 48000)
        self.wrapper.diffusion_scheduler.step.return_value = torch.zeros(1, 1, 48000)
        self.wrapper.diffusion_scheduler.step_ddim.return_value = torch.zeros(1, 1, 48000)
        # Mock add_noise to return (x, noise)
        self.wrapper.diffusion_scheduler.add_noise.side_effect = lambda x, t, noise=None: (x, torch.randn_like(x))
        
        self.wrapper.model = MagicMock()
        self.wrapper.model.return_value = torch.randn(1, 1, 48000) # Dummy noise output

    def test_ddim_called_for_strided_sampling(self):
        """Verify step_ddim is used when skipping steps"""
        # 25 steps vs 1000 total -> stride 40
        waveform = torch.randn(1, 48000) # 1 sec
        
        # Mock estimate_max_batch_size
        self.wrapper.estimate_max_batch_size = MagicMock(return_value=1)
        
        self.wrapper.enhance(
            waveform, 
            original_sr=48000, 
            target_sr=48000, 
            chunk_seconds=2.0,
            diffusion_steps=25
        )
        
        # Should call step_ddim
        self.wrapper.diffusion_scheduler.step_ddim.assert_called()
        # Should NOT call standard step
        self.wrapper.diffusion_scheduler.step.assert_not_called()
        print("Verified: step_ddim called for 25 steps")

    def test_ddpm_called_for_full_sampling(self):
        """Verify standard step is used for full sampling"""
        # 1000 steps vs 1000 total -> stride 1
        waveform = torch.randn(1, 48000)
        
        self.wrapper.estimate_max_batch_size = MagicMock(return_value=1)
        
        self.wrapper.enhance(
            waveform, 
            original_sr=48000, 
            target_sr=48000, 
            chunk_seconds=2.0,
            diffusion_steps=1000
        )
        
        # Should call standard step
        self.wrapper.diffusion_scheduler.step.assert_called()
        # Should NOT call step_ddim
        self.wrapper.diffusion_scheduler.step_ddim.assert_not_called()
        print("Verified: standard step called for 1000 steps")

    def test_ddim_final_step_handling(self):
        """Verify step_ddim handles t_prev=-1 without crashing"""
        # Create a real scheduler instance to test the actual step_ddim logic
        from ai_audio_upscaler.ai_upscaler.diffusion import DiffusionScheduler
        scheduler = DiffusionScheduler(num_steps=100)
        
        # Inputs
        batch_size = 2
        model_output = torch.randn(batch_size, 1, 100)
        t = torch.tensor([10, 10]) # Current step
        x_t = torch.randn(batch_size, 1, 100)
        t_prev = torch.tensor([-1, -1]) # Final step (next is -1)
        
        # This should not raise RuntimeError
        try:
            x_prev = scheduler.step_ddim(model_output, t, x_t, t_prev)
            print("Verified: step_ddim handled t_prev=-1 successfully")
            self.assertEqual(x_prev.shape, x_t.shape)
        except RuntimeError as e:
            self.fail(f"step_ddim raised RuntimeError with t_prev=-1: {e}")

if __name__ == '__main__':
    unittest.main()
