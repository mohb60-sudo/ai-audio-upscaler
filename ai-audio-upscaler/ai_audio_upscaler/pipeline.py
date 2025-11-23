import torch
import torchaudio
import logging
import os
from .config import UpscalerConfig
from .dsp import DSPUpscaler
# AI import will be inside the method to avoid circular deps or early load issues if not needed
# from .ai_upscaler.inference import AIUpscalerWrapper

logger = logging.getLogger(__name__)

class AudioUpscalerPipeline:
    """
    Orchestrates the upscaling process: loading, processing (DSP or AI), and saving.
    """
    def __init__(self, config: UpscalerConfig):
        self.config = config
        self.dsp = DSPUpscaler(config.target_sample_rate, config.baseline_method)
        
        self.ai_model = None
        if self.config.mode == "ai":
            from .ai_upscaler.inference import AIUpscalerWrapper
            self.ai_model = AIUpscalerWrapper(config)

    def load_audio(self, file_path: str):
        """Loads audio file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading audio: {file_path}")
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            raise e
        except Exception as e:
            # Fallback: Try to use FFmpeg CLI to convert to WAV
            # This handles cases where torchaudio backend (soundfile) doesn't support the format (M4A/MP3)
            # but FFmpeg is installed on the system.
            import subprocess
            import tempfile
            import shutil
            
            if shutil.which("ffmpeg"):
                logger.info(f"Torchaudio failed to load {file_path}. Attempting FFmpeg CLI conversion...")
                try:
                    # Create temp WAV file
                    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    
                    # Convert: ffmpeg -i input -y output.wav
                    subprocess.run(
                        ["ffmpeg", "-i", file_path, "-y", temp_wav], 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    
                    # Load the temp WAV
                    waveform, sample_rate = torchaudio.load(temp_wav)
                    
                    # Cleanup
                    os.remove(temp_wav)
                    return waveform, sample_rate
                    
                except Exception as conversion_error:
                    logger.error(f"FFmpeg CLI conversion failed: {conversion_error}")
                    # If conversion fails, raise the original error or the new one
                    pass
            
            # If we reach here, fallback failed.
            # Check for common "Format not recognised" error from soundfile/libsndfile
            if "Format not recognised" in str(e):
                raise ValueError(
                    f"Error loading '{os.path.basename(file_path)}'. "
                    "This format (likely M4A/AAC/MP3) requires FFmpeg, which was not found on your system.\n"
                    "Please install FFmpeg or convert the file to WAV."
                ) from e
            raise e
        return waveform, sample_rate

    def save_audio(self, waveform: torch.Tensor, file_path: str, sample_rate: int):
        """Saves audio file."""
        logger.info(f"Saving audio to: {file_path} at {sample_rate} Hz")
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format from config or extension
        format = self.config.export_format.lower()
        
        # Override extension if needed
        base, _ = os.path.splitext(file_path)
        if not file_path.lower().endswith(f".{format}"):
            file_path = f"{base}.{format}"
            
        try:
            if format == "mp3":
                # Torchaudio/soundfile doesn't always support MP3 writing easily on all platforms
                # We use the same fallback strategy as loading if needed, but let's try direct first
                torchaudio.save(file_path, waveform, sample_rate, format="mp3")
            elif format == "flac":
                torchaudio.save(file_path, waveform, sample_rate, format="flac")
            elif format == "ogg":
                torchaudio.save(file_path, waveform, sample_rate, format="ogg")
            else:
                # Default to WAV (32-bit float)
                torchaudio.save(file_path, waveform, sample_rate, bits_per_sample=32)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            # Fallback to WAV if specific format fails
            fallback_path = f"{base}.wav"
            logger.info(f"Falling back to WAV: {fallback_path}")
            torchaudio.save(fallback_path, waveform, sample_rate, bits_per_sample=32)

    def run(self, input_path: str, output_path: str, normalize: bool = False, generate_analysis: bool = False, progress_callback=None):
        """
        Runs the full pipeline on a file.
        
        Args:
            input_path: Path to input audio.
            output_path: Path to save output.
            normalize: Whether to normalize output to -1.0 dB.
            generate_analysis: Whether to return spectrogram figures.
            progress_callback: Optional function accepting (progress: float, message: str).
        
        Returns:
            Dictionary containing paths and optional analysis figures.
        """
        if progress_callback: progress_callback(0.1, "Loading audio...")
        waveform, sr = self.load_audio(input_path)
        
        # Store input for analysis if needed
        input_waveform = waveform if generate_analysis else None
        input_sr = sr
        
        # 1. Baseline Resampling
        if progress_callback: progress_callback(0.3, f"Resampling ({self.config.baseline_method})...")
        upsampled_waveform = self.dsp.process(waveform, sr)
        
        final_waveform = upsampled_waveform

        # 2. AI Enhancement (Optional)
        if self.config.mode == "ai" and self.ai_model:
            if progress_callback: progress_callback(0.5, "Applying AI enhancement (this may take a while)...")
            logger.info("Applying AI enhancement...")
            final_waveform = self.ai_model.enhance(upsampled_waveform, sr, self.config.target_sample_rate)
        
        # 3. Normalization
        if normalize:
            if progress_callback: progress_callback(0.8, "Normalizing audio...")
            logger.info("Normalizing output audio...")
            max_val = torch.max(torch.abs(final_waveform))
            if max_val > 0:
                target_db = -1.0
                target_amp = 10 ** (target_db / 20)
                final_waveform = final_waveform * (target_amp / max_val)
        
        # Safety Limiter: Ensure audio never exceeds [-1, 1] to prevent clipping
        # This runs regardless of whether normalization was applied or not
        max_peak = torch.max(torch.abs(final_waveform))
        if max_peak > 1.0:
            logger.warning(f"Audio clipping detected (Peak: {max_peak:.2f}). Clamping to [-1.0, 1.0].")
            final_waveform = torch.clamp(final_waveform, -1.0, 1.0)

        if progress_callback: progress_callback(0.9, "Saving output...")
        self.save_audio(final_waveform, output_path, self.config.target_sample_rate)
        
        results = {"output_path": output_path}
        
        # 4. Analysis
        if generate_analysis:
            if progress_callback: progress_callback(0.95, "Generating spectrograms...")
            from .analysis import generate_spectrogram
            
            # Input Spectrogram
            fig_in = generate_spectrogram(input_waveform, input_sr, title=f"Input ({input_sr} Hz)")
            results["input_spectrogram"] = fig_in
            
            # Output Spectrogram
            fig_out = generate_spectrogram(final_waveform, self.config.target_sample_rate, title=f"Output ({self.config.target_sample_rate} Hz)")
            results["output_spectrogram"] = fig_out
            
        if progress_callback: progress_callback(1.0, "Done!")
        logger.info("Processing complete.")
        return results
