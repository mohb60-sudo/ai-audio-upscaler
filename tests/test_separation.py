import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock demucs BEFORE importing separation
mock_demucs = MagicMock()
sys.modules["demucs"] = mock_demucs
sys.modules["demucs.pretrained"] = mock_demucs.pretrained
sys.modules["demucs.apply"] = mock_demucs.apply

from ai_audio_upscaler.separation import SourceSeparator

class TestSourceSeparator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        
    @patch("ai_audio_upscaler.separation.pretrained")
    @patch("ai_audio_upscaler.separation.apply_model")
    def test_separate_flow(self, mock_apply, mock_pretrained):
        """Verifies that separate() calls Demucs correctly and returns expected dict."""
        
        # Mock Model
        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_model.sources = ["drums", "bass", "other", "vocals"]
        mock_pretrained.get_model.return_value = mock_model
        
        # Mock Apply Output: (Batch, Sources, Channels, Time)
        # 1 Batch, 4 Sources, 2 Channels, 1000 Samples
        mock_output = torch.randn(1, 4, 2, 1000)
        mock_apply.return_value = mock_output
        
        separator = SourceSeparator(self.device)
        
        # Input: 2 Channels, 1000 Samples, 44100 Hz
        waveform = torch.randn(2, 1000)
        sr = 44100
        
        stems = separator.separate(waveform, sr)
        
        # Verify Keys
        self.assertEqual(list(stems.keys()), ["drums", "bass", "other", "vocals"])
        
        # Verify Shapes
        for name, stem in stems.items():
            self.assertEqual(stem.shape, (2, 1000))
            
        # Verify Calls
        mock_pretrained.get_model.assert_called_once()
        mock_apply.assert_called_once()

    @patch("ai_audio_upscaler.separation.pretrained")
    @patch("ai_audio_upscaler.separation.apply_model")
    def test_resampling(self, mock_apply, mock_pretrained):
        """Verifies that separate() handles resampling."""
        
        # Mock Model (44.1k)
        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_model.sources = ["drums", "bass", "other", "vocals"]
        mock_pretrained.get_model.return_value = mock_model
        
        # Mock Output (at 44.1k)
        # Input is 48k. 1000 samples at 48k -> ~918 samples at 44.1k
        mock_output = torch.randn(1, 4, 2, 918)
        mock_apply.return_value = mock_output
        
        separator = SourceSeparator(self.device)
        
        # Input: 48k
        waveform = torch.randn(2, 1000)
        sr = 48000
        
        stems = separator.separate(waveform, sr)
        
        # Verify Output Shape (Should be back to 1000)
        for name, stem in stems.items():
            self.assertEqual(stem.shape, (2, 1000))

if __name__ == "__main__":
    unittest.main()
