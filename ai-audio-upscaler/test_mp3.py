import torch
import torchaudio
import io
import os

def test_mp3_encoding():
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Create dummy audio
    sr = 44100
    waveform = torch.randn(1, sr) # 1 second of noise
    
    # Try saving to bytes buffer
    buffer = io.BytesIO()
    try:
        torchaudio.save(buffer, waveform, sr, format="mp3")
        print("Successfully saved to MP3 buffer.")
        
        buffer.seek(0)
        loaded_waveform, loaded_sr = torchaudio.load(buffer, format="mp3")
        print(f"Successfully loaded from MP3 buffer. Shape: {loaded_waveform.shape}, SR: {loaded_sr}")
        
    except Exception as e:
        print(f"Failed to save/load MP3: {e}")

if __name__ == "__main__":
    test_mp3_encoding()
