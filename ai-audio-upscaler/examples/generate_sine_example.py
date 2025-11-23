import torch
import torchaudio
import numpy as np
import os

def generate_sine_wave(filename="test_sine.wav", sample_rate=16000, duration=3.0, freq=440.0):
    """Generates a sine wave audio file."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * freq * t)
    
    # Convert to tensor and add channel dim
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    
    print(f"Generating {filename} at {sample_rate} Hz")
    torchaudio.save(filename, waveform_tensor, sample_rate)

if __name__ == "__main__":
    os.makedirs("examples", exist_ok=True)
    generate_sine_wave("examples/input_16k.wav", sample_rate=16000)
