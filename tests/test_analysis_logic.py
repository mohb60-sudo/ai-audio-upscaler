import torch
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_audio_upscaler.analysis import detect_cutoff_frequency, estimate_noise_floor, count_clipping, calculate_lra

def test_cutoff_detection():
    print("Testing Cutoff Detection...")
    sr = 96000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # 1. Full Bandwidth White Noise
    full_bw = torch.rand(1, int(sr * duration)) * 2 - 1
    cutoff_full = detect_cutoff_frequency(full_bw, sr)
    print(f"  Full Bandwidth (Expected > 40kHz): {cutoff_full:.1f} Hz")
    
    # 2. Fake High-Res (Low Pass at 20kHz)
    # Simple brick-wall approximation in frequency domain
    n_fft = 4096
    window = torch.hann_window(n_fft)
    stft = torch.stft(full_bw, n_fft=n_fft, hop_length=1024, window=window, return_complex=True)
    freqs = torch.linspace(0, sr/2, n_fft//2 + 1)
    mask = freqs <= 20000
    stft[:, ~mask, :] = 0 # Kill everything above 20kHz
    fake_bw = torch.istft(stft, n_fft=n_fft, hop_length=1024, window=window, length=full_bw.shape[1])
    
    cutoff_fake = detect_cutoff_frequency(fake_bw, sr)
    print(f"  Fake High-Res (Expected ~20kHz): {cutoff_fake:.1f} Hz")
    
    assert cutoff_full > 40000, "Failed to detect full bandwidth"
    assert 18000 < cutoff_fake < 22000, f"Failed to detect 20kHz cutoff (Got {cutoff_fake:.1f} Hz)"
    print("✅ Cutoff Detection Passed")

def test_clipping_detection():
    print("\nTesting Clipping Detection...")
    sr = 44100
    t = torch.linspace(0, 1, sr)
    # Sine wave amplitude 1.5, clamped to 1.0
    sine = torch.sin(2 * 3.14159 * 440 * t) * 1.5
    clipped = torch.clamp(sine, -1.0, 1.0)
    
    count = count_clipping(clipped.unsqueeze(0))
    print(f"  Clipped Samples (Expected > 0): {count}")
    
    assert count > 1000, "Failed to detect clipping"
    print("✅ Clipping Detection Passed")

def test_noise_floor():
    print("\nTesting Noise Floor...")
    sr = 44100
    # Silence with tiny noise
    silence = torch.randn(1, sr) * 0.0001 # -80dB
    noise_floor = estimate_noise_floor(silence)
    print(f"  Noise Floor (Expected ~ -80dB): {noise_floor}")
    
    # Parse string "-80.0 dB"
    val = float(noise_floor.split()[0])
    assert -90 < val < -70, f"Noise floor estimate off (Got {val})"
    print("✅ Noise Floor Passed")

if __name__ == "__main__":
    try:
        test_cutoff_detection()
        test_clipping_detection()
        test_noise_floor()
        print("\n🎉 ALL ANALYSIS TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
