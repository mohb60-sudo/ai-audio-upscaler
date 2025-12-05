import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.safety import GPUGuard

class TestGPUSafety(unittest.TestCase):
    def setUp(self):
        self.guard = GPUGuard(device="cpu", watchdog=False)
        
    def test_init(self):
        """Test initialization."""
        self.assertFalse(self.guard.watchdog_enabled)
        self.assertEqual(self.guard.max_kernel_ms, 1200)
        
    def test_auto_batch_size_cpu(self):
        """Test auto-batch fallback on CPU."""
        # Should return max_bs or default
        bs = self.guard.auto_batch_size(lambda x: True, max_bs=16)
        self.assertEqual(bs, 16)
        
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_reserved', return_value=0)
    @patch('torch.cuda.memory_allocated', return_value=0)
    def test_auto_batch_size_cuda(self, mock_alloc, mock_res, mock_props, mock_cuda):
        """Test binary search logic."""
        mock_props.return_value.total_memory = 10 * 1024**3 # 10GB
        
        # Mock try_fn: fails if bs > 32
        def try_fn(bs):
            if bs > 32:
                raise RuntimeError("CUDA out of memory")
            return True
            
        bs = self.guard.auto_batch_size(try_fn, max_bs=64)
        self.assertEqual(bs, 32)
        
    def test_safe_step_retry(self):
        """Test retry logic on OOM."""
        mock_step = MagicMock()
        # Fail twice with OOM, then succeed
        mock_step.side_effect = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory"),
            "Success"
        ]
        
        result = self.guard.safe_step(mock_step, retries=3)
        self.assertEqual(result, "Success")
        self.assertEqual(mock_step.call_count, 3)

    def test_safe_step_fail(self):
        """Test failure after max retries."""
        mock_step = MagicMock()
        mock_step.side_effect = RuntimeError("CUDA out of memory")
        
        with self.assertRaises(RuntimeError):
            self.guard.safe_step(mock_step, retries=2)
            
        self.assertEqual(mock_step.call_count, 3) # Initial + 2 retries

    @patch('psutil.virtual_memory')
    def test_optimize_workers(self, mock_mem):
        """Test worker optimization based on RAM."""
        # Mock 8GB total, 4GB available
        mock_mem.return_value.available = 4 * 1024 * 1024 * 1024
        
        # Request 10 workers
        # Safe RAM = 4GB - 2GB = 2GB
        # Cost per worker = 100MB + 100MB = 200MB
        # Max workers = 2048 / 200 = 10
        
        # Case 1: Enough RAM
        safe = self.guard.optimize_workers(8, avg_file_size_mb=50)
        self.assertEqual(safe, 8)
        
        # Case 2: Low RAM (1GB available)
        mock_mem.return_value.available = 1 * 1024 * 1024 * 1024
        # Safe RAM = 0 (Clamped) -> Max workers = 0
        safe = self.guard.optimize_workers(4, avg_file_size_mb=50)
        self.assertEqual(safe, 0)

    @patch('psutil.virtual_memory')
    def test_check_system_ram(self, mock_mem):
        """Test RAM threshold check."""
        mock_mem.return_value.percent = 95
        self.assertTrue(self.guard.check_system_ram(threshold_percent=90))
        
        mock_mem.return_value.percent = 80
        self.assertFalse(self.guard.check_system_ram(threshold_percent=90))

if __name__ == '__main__':
    unittest.main()
