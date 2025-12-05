import torch
import unittest
from ai_audio_upscaler.ai_upscaler.diffusion import DiffusionScheduler

class TestDiffusionScheduler(unittest.TestCase):
    def test_step_batch_support(self):
        """Verify step() handles batched inputs correctly"""
        scheduler = DiffusionScheduler(num_steps=1000)
        
        batch_size = 4
        channels = 1
        time_dim = 100
        
        # Create batched inputs
        x_t = torch.randn(batch_size, channels, time_dim)
        model_output = torch.randn(batch_size, channels, time_dim)
        
        # Create batched timesteps (all same or different)
        t = torch.full((batch_size,), 500, dtype=torch.long)
        
        # Run step
        try:
            x_prev = scheduler.step(model_output, t, x_t)
            
            # Check output shape
            self.assertEqual(x_prev.shape, (batch_size, channels, time_dim))
            print("Batch step successful!")
            
        except RuntimeError as e:
            self.fail(f"step() failed with RuntimeError: {e}")

if __name__ == '__main__':
    unittest.main()
