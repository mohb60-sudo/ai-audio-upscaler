import os
import torch
import torchaudio
import subprocess
import uuid
from typing import Tuple

def load_audio_robust(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Robust audio loader that falls back to FFmpeg for unsupported formats (like M4A on Windows).
    
    Args:
        file_path (str): Absolute path to the audio file.
        
    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - waveform (torch.Tensor): Audio data (Channels, Time).
            - sample_rate (int): Sample rate in Hz.
            
    Raises:
        RuntimeError: If both direct load and FFmpeg fallback fail.
    """
    try:
        return torchaudio.load(file_path)
    except Exception as e:
        # Fallback to FFmpeg
        
        # Create temp filename in the same directory to avoid permission issues with system temp
        temp_dir = os.path.dirname(file_path)
        temp_wav = os.path.join(temp_dir, f"temp_convert_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            # Convert to WAV using FFmpeg
            # -y: overwrite
            # -i: input
            # -vn: disable video (just in case)
            # -acodec pcm_f32le: preserve quality (32-bit float)
            subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, "-vn", "-acodec", "pcm_f32le", temp_wav], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            waveform, sr = torchaudio.load(temp_wav)
            return waveform, sr
            
        except Exception as ffmpeg_error:
            raise RuntimeError(f"Failed to load audio: {e}. FFmpeg fallback failed: {ffmpeg_error}")
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass
