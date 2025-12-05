import torch
import unittest
from unittest.mock import MagicMock, patch
from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from ai_audio_upscaler.config import UpscalerConfig

class TestInferenceProgress(unittest.TestCase):
    def setUp(self):
        self.config = UpscalerConfig(
            target_sample_rate=48000,
            mode="ai",
            device="cpu", # Use CPU for test speed
            model_checkpoint=None
        )
        
        # Mock the wrapper setup to avoid loading real models
        with patch("ai_audio_upscaler.ai_upscaler.inference.AudioSuperResNet"), \
             patch("ai_audio_upscaler.ai_upscaler.inference.DiffusionUNet"), \
             patch("ai_audio_upscaler.ai_upscaler.inference.DiffusionScheduler"):
            self.wrapper = AIUpscalerWrapper(self.config)
            
        # Manually setup mocks
        self.wrapper.use_diffusion = True
        self.wrapper.model = MagicMock()
        # Ensure model returns a tensor of same shape as input
        self.wrapper.model.side_effect = lambda x, *args, **kwargs: torch.randn_like(x)
        self.wrapper.diffusion_scheduler = MagicMock()
        self.wrapper.diffusion_scheduler.num_steps = 1000
        # Mock step to return same shape as input (simulating correct batch handling)
        self.wrapper.diffusion_scheduler.step.side_effect = lambda pred, t, x: x
        self.wrapper.diffusion_scheduler.step_ddim.side_effect = lambda pred, t, x, t_prev, eta=0.0, **kwargs: x
        # Mock add_noise to return (noisy_x, noise)
        def mock_add_noise(*args, **kwargs):
            # args[0] is x, args[1] is t
            x = args[0]
            noise = kwargs.get('noise', args[2] if len(args) > 2 else None)
            if noise is None:
                noise = torch.randn_like(x)
            return x, noise
        self.wrapper.diffusion_scheduler.add_noise.side_effect = mock_add_noise
        self.wrapper.device = torch.device("cpu")
        
        # Mock estimate_max_batch_size to allow batching
        self.wrapper.estimate_max_batch_size = MagicMock(return_value=8) # Allow 8 items

    def test_input_clamping(self):
        """Verify that input waveform is clamped to [-1, 1]"""
        # Create waveform with values outside [-1, 1]
        sr = 48000
        waveform = torch.randn(1, sr * 2) * 5.0 # High amplitude noise
        
        # Mock _enhance_batch to check its input
        with patch.object(self.wrapper, '_enhance_batch') as mock_enhance_batch:
            mock_enhance_batch.return_value = torch.zeros_like(waveform)
            
            self.wrapper.enhance(
                waveform, 
                original_sr=sr, 
                target_sr=sr, 
                chunk_seconds=2.0,
                diffusion_steps=5
            )
            
            # Check arguments passed to _enhance_batch
            args, _ = mock_enhance_batch.call_args
            processed_waveform = args[0] # First arg is waveform (batched)
            
            # Verify max/min values
            self.assertLessEqual(processed_waveform.max().item(), 1.0001)
            self.assertGreaterEqual(processed_waveform.min().item(), -1.0001)
            print("Input clamping verified!")

    def test_progress_callback_called(self):
        """Verify progress callback is called during inference"""
        # Create dummy waveform: 1 channel, 4 seconds (2 chunks of 2s)
        sr = 48000
        waveform = torch.randn(1, sr * 4) 
        
        mock_callback = MagicMock()
        
        # Run enhance
        self.wrapper.enhance(
            waveform, 
            original_sr=sr, 
            target_sr=sr, 
            chunk_seconds=2.0,
            diffusion_steps=10, # Low steps for speed
            progress_callback=mock_callback
        )
        
        # Verify callback was called
        self.assertTrue(mock_callback.called)
        
        # Check arguments of first call
        args, _ = mock_callback.call_args_list[0]
        progress, msg = args
        print(f"Callback called with: {progress}, {msg}")
        self.assertGreaterEqual(progress, 0.5)
        self.assertIn("Diffusion:", msg)

    def test_batching_logic(self):
        """Verify that batching processes multiple chunks at once"""
        # Create dummy waveform: 1 channel, 8 seconds (4 chunks of 2s)
        sr = 48000
        waveform = torch.randn(1, sr * 8)
        
        # wrapper.estimate_max_batch_size returns 8, so batch_size should be 8/1 = 8
        # We have 4 chunks, so they should all fit in 1 batch.
        
        self.wrapper.model.reset_mock()
        
        self.wrapper.enhance(
            waveform, 
            original_sr=sr, 
            target_sr=sr, 
            chunk_seconds=2.0,
            diffusion_steps=5,
            denoising_strength=1.0
        )
        
        # Verify model was called. 
        # With 5 steps, model should be called 5 times.
        # Each call should have a batch size of 4 (4 chunks).
        
        self.assertEqual(self.wrapper.model.call_count, 5)
        
        # Check input shape of first call
        # args: (x, t), kwargs: {condition: ...}
        args, kwargs = self.wrapper.model.call_args
        x, t = args
        condition = kwargs.get('condition')
        
        # x shape should be (Batch*Channels, 1, Time)
        # Batch=4 chunks, Channels=1. So (4, 1, Time)
        print(f"Model input shape: {x.shape}")
        self.assertEqual(x.shape[0], 4) 

if __name__ == '__main__':
    unittest.main()
