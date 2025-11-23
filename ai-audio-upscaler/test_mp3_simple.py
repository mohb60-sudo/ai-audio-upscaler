import torch
import torchaudio
import io

def test_mp3_soundfile():
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Backends: {torchaudio.list_audio_backends()}")
    
    sr = 44100
    waveform = torch.randn(1, sr)
    buffer = io.BytesIO()
    
    try:
        # Try without compression arg
        torchaudio.save(buffer, waveform, sr, format="mp3")
        print("Success: Saved MP3 without compression arg.")
        print(f"Buffer size: {buffer.tell()} bytes")
        
    except Exception as e:
        print(f"Failed without compression arg: {e}")

if __name__ == "__main__":
    test_mp3_soundfile()
