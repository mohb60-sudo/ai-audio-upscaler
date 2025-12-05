import torch
import torchaudio
import os
import sys
import subprocess
import shutil
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.audio_io import load_audio_robust
from train import find_audio_files

def test_m4a_loading():
    print("Testing M4A Support (Loading & Discovery)...")
    
    # 1. Setup paths
    test_dir = "tests/temp_audio"
    os.makedirs(test_dir, exist_ok=True)
    wav_path = os.path.join(test_dir, "test_tone.wav")
    m4a_path = os.path.join(test_dir, "test_tone.m4a")
    
    try:
        # 2. Generate Synthetic Audio (Sine Wave)
        sr = 48000
        duration = 1.0
        t = torch.linspace(0, duration, int(sr * duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) # 440Hz A4
        
        # Save as WAV first
        torchaudio.save(wav_path, waveform, sr)
        
        # 3. Convert to M4A using FFmpeg
        if not shutil.which("ffmpeg"):
            print("⚠️ FFmpeg not found. Skipping M4A test.")
            return
            
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "192k", m4a_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 4. Verify Discovery
        print(f"  Scanning {test_dir}...")
        found_files = find_audio_files(test_dir)
        print(f"  Found: {found_files}")
        
        assert any(f.endswith(".m4a") for f in found_files), "M4A file not found by discovery logic"
        
        # 5. Try to load M4A using our robust loader
        print(f"  Loading {m4a_path}...")
        loaded_wave, loaded_sr = load_audio_robust(m4a_path)
        
        # 5. Verify
        print(f"  Loaded Shape: {loaded_wave.shape}")
        print(f"  Loaded SR: {loaded_sr}")
        
        assert loaded_sr == sr, f"Sample rate mismatch: {loaded_sr} != {sr}"
        assert loaded_wave.shape[1] > 0, "Loaded waveform is empty"
        
        # Check correlation/similarity (M4A is lossy, so exact match isn't expected)
        # We assume the length might slightly differ due to padding/priming
        min_len = min(waveform.shape[1], loaded_wave.shape[1])
        
        # Normalize for comparison
        orig_seg = waveform[0, :min_len]
        loaded_seg = loaded_wave[0, :min_len]
        
        # Simple correlation
        correlation = torch.corrcoef(torch.stack([orig_seg, loaded_seg]))[0, 1]
        print(f"  Correlation with original: {correlation:.4f}")
        
        assert correlation > 0.95, f"Correlation too low ({correlation:.4f}). Loading might be garbage."
        
        print("✅ M4A Loading Passed")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_m4a_loading()
