import torch
import torchaudio
import logging
import os
from .config import UpscalerConfig
from .dsp_basic import DSPUpscaler
from .dsp_advanced import AdvancedDSPUpscaler, auto_select_filter
from .dsp.chain import AudioChain
from ai_audio_upscaler.audio_io import load_audio_robust

logger = logging.getLogger(__name__)

class AudioUpscalerPipeline:
    """
    Orchestrates the complete audio upscaling workflow.
    
    Responsibilities:
    1. Loading audio from various formats (WAV, FLAC, MP3, etc.)
    2. Selecting and initializing the appropriate DSP engine (Basic vs Advanced)
    3. Managing AI model inference if enabled
    4. Normalizing output levels
    5. Generating analysis data (Spectrograms, PSD)
    6. Saving the processed result
    """
    def __init__(self, config: UpscalerConfig):
        self.config = config
        
        # Select DSP upscaler based on configuration
        if config.use_advanced_dsp or config.baseline_method in ["poly-sinc", "poly-sinc-hq", "poly-sinc-fast"]:
            # Use advanced DSP with HQPlayer-inspired filters
            logger.info(f"Using Advanced DSP: {config.baseline_method}")
            
            # Auto-select filter settings if using preset
            if config.noise_shaper == "auto":
                filter_settings = auto_select_filter(
                    input_sr=44100,  # Dummy value for init, will be handled by DSP module
                    output_sr=config.target_sample_rate,
                    mode=config.dsp_quality_preset
                )
                filter_type = filter_settings["filter_type"]
                dither = filter_settings["dither"] and config.apply_dither
                noise_shaper = filter_settings["noise_shaper"]
            else:
                filter_type =config.baseline_method if config.baseline_method.startswith("poly-sinc") else "poly-sinc"
                dither = config.apply_dither
                noise_shaper = config.noise_shaper
            
            self.dsp = AdvancedDSPUpscaler(
                target_sample_rate=config.target_sample_rate,
                filter_type=filter_type,
                dither=dither,
                noise_shaper=noise_shaper
            )
        else:
            # Use basic DSP
            logger.info(f"Using Basic DSP: {config.baseline_method}")
            self.dsp = DSPUpscaler(config.target_sample_rate, config.baseline_method)
        
        self.ai_model = None
        if self.config.mode == "ai":
            from .ai_upscaler.inference import AIUpscalerWrapper
            self.ai_model = AIUpscalerWrapper(config)

        # Initialize Professional Audio Chain
        self.chain = AudioChain(config.target_sample_rate)

    def load_audio(self, file_path: str):
        """
        Robustly loads audio files with fallback mechanisms.
        Delegates to ai_audio_upscaler.audio_io.load_audio_robust.
        
        Args:
            file_path (str): Absolute path to the audio file
            
        Returns:
            tuple: (waveform_tensor, sample_rate_int)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading audio: {file_path}")
        return load_audio_robust(file_path)

    def save_audio(self, waveform: torch.Tensor, file_path: str, sample_rate: int):
        """Saves audio file."""
        logger.info(f"Saving audio to: {file_path} at {sample_rate} Hz")
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format from config or extension
        format = self.config.export_format.lower()
        
        base, _ = os.path.splitext(file_path)
        if not file_path.lower().endswith(f".{format}"):
            file_path = f"{base}.{format}"
            
        try:
            if format == "mp3":
                # Torchaudio/soundfile doesn't always support MP3 writing easily on all platforms
                # We use the same fallback strategy as loading if needed, but let's try direct first
                torchaudio.save(file_path, waveform, sample_rate, format="mp3")
            elif format == "flac":
                # User requested 24-bit FLAC
                torchaudio.save(file_path, waveform, sample_rate, format="flac", bits_per_sample=24)
            elif format == "ogg":
                torchaudio.save(file_path, waveform, sample_rate, format="ogg")
            else:
                # Default to WAV (24-bit PCM as requested)
                torchaudio.save(file_path, waveform, sample_rate, bits_per_sample=24)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            # Fallback to WAV if specific format fails
            fallback_path = f"{base}.wav"
            logger.info(f"Falling back to WAV: {fallback_path}")
            torchaudio.save(fallback_path, waveform, sample_rate, bits_per_sample=24)

    def run(self, input_path: str, output_path: str, normalization_mode: str = "Peak -1dB", generate_analysis: bool = False, progress_callback=None,
            tta: bool = False, stereo_mode: str = "lr", transient_strength: float = 0.0, spectral_matching: bool = False,
            remove_dc: bool = True, normalize_input: bool = True, limit: bool = True, saturation: float = 0.0, stereo_width: float = 1.0, transient_shaping: float = 0.0,
            qc: bool = False, candidate_count: int = 8, judge_threshold: float = 0.5,
            denoising_strength: float = 0.6, use_stems: bool = False,
            use_restoration: bool = False, restoration_strength: float = 0.5):
        """
        Runs the full pipeline on a file.
        
        Args:
            input_path: Path to input audio.
            output_path: Path to save output.
            normalization_mode: "Match Source", "Peak -1dB", "Peak -3dB", or "None".
            generate_analysis: Whether to return spectrogram figures.
            progress_callback: Optional function accepting (progress: float, message: str).
            # Post-Processing Args (Legacy & New)
            tta: bool = False,
            stereo_mode: str = "lr",
            transient_strength: float = 0.0, # Original restoration
            spectral_matching: bool = False,
            # Pro Mastering Args
            remove_dc: bool = True,
            normalize_input: bool = True,
            limit: bool = True,
            saturation: float = 0.0,
            stereo_width: float = 1.0,
            transient_shaping: float = 0.0, # Mastering compressor
            # Quality Control Args
            qc: bool = False,
            candidate_count: int = 8,
            judge_threshold: float = 0.5
            denoising_strength: float = 0.6
        
        Returns:
            Dictionary containing paths and optional analysis figures.
        """
        if progress_callback:
            progress_callback(0.1, "Loading audio...")
        waveform, sr = self.load_audio(input_path)
        
        # Store input for analysis if needed
        input_waveform = waveform if generate_analysis else None
        input_sr = sr
        
        # --- 0. Pre-Processing (Conditioning) ---
        if progress_callback:
            progress_callback(0.2, "Conditioning audio (DC Offset, LUFS)...")
        
        # Pre-processing (Conditioning)
        # We process at input SR before upsampling to fix DC offset and loudness.
        
        # Save original waveform for "Match Source" normalization
        original_waveform = waveform.clone()
        
        # Pre-process (Normalize Input to -18 LUFS for consistent AI processing)
        waveform, input_gain_db = self.chain.pre_process(waveform, sr=input_sr, remove_dc=remove_dc, normalize_lufs=normalize_input)
        logger.info(f"Pre-processing applied. Input Gain: {input_gain_db:.2f} dB")

        # 1. Baseline Resampling
        if progress_callback:
            progress_callback(0.3, f"Resampling ({self.config.baseline_method})...")
        upsampled_waveform = self.dsp.process(waveform, sr)
        
        final_waveform = upsampled_waveform

        # 2. AI Enhancement (Optional)
        if self.config.mode == "ai" and self.ai_model:
            if progress_callback:
                progress_callback(0.5, "Applying AI enhancement (this may take a while)...")
            logger.info("Applying AI enhancement...")
            logger.info("Applying AI enhancement...")
            final_waveform = self.ai_model.enhance(
                upsampled_waveform, 
                sr, 
                self.config.target_sample_rate,
                tta=tta,
                stereo_mode=stereo_mode,
                transient_strength=transient_strength,
                spectral_matching=spectral_matching,
                qc=qc,
                candidate_count=candidate_count,
                judge_threshold=judge_threshold,
                denoising_strength=denoising_strength,
                diffusion_steps=self.config.diffusion_steps,
                progress_callback=progress_callback,
                use_stems=use_stems,
                use_restoration=use_restoration,
                restoration_strength=restoration_strength
            )

        
        # 3. Mastering Chain (Post-Processing)
        if progress_callback:
            progress_callback(0.8, "Applying Mastering Chain...")
            
        final_waveform = self.chain.post_process(
            final_waveform, 
            limit=limit, 
            saturation=saturation, 
            stereo_width=stereo_width, 
            transient_shaping=transient_shaping,
            dither=self.config.apply_dither
        )

        # --- Flexible Normalization ---
        # --- Flexible Normalization ---
        if normalization_mode == "Match Source":
            logger.info("Applying Normalization: Match Source Loudness")
            
            # We measure the integrated loudness (LUFS) of the source and target
            # and adjust the target gain to match the source.
            import pyloudnorm as pyln
            import numpy as np
            
            try:
                # Measure Source Loudness (at input SR)
                meter_src = pyln.Meter(input_sr)
                src_np = original_waveform.cpu().t().numpy()
                src_lufs = meter_src.integrated_loudness(src_np)
                
                # Measure Target Loudness (at target SR)
                meter_tgt = pyln.Meter(self.config.target_sample_rate)
                tgt_np = final_waveform.cpu().t().numpy()
                tgt_lufs = meter_tgt.integrated_loudness(tgt_np)
                
                if not (np.isinf(src_lufs) or np.isinf(tgt_lufs)):
                    delta = src_lufs - tgt_lufs
                    gain_linear = 10.0 ** (delta / 20.0)
                    final_waveform = final_waveform * gain_linear
                    logger.info(f"Matched Loudness: Source {src_lufs:.1f} LUFS -> Target {tgt_lufs:.1f} LUFS (Delta: {delta:+.1f} dB)")
                else:
                    logger.warning("Loudness measurement returned -inf (silence?). Skipping match.")
                    
            except Exception as e:
                logger.warning(f"Loudness matching failed: {e}")
                
        elif normalization_mode == "Peak -1dB":
            logger.info("Applying Normalization: Peak -1.0 dB")
            final_waveform = self.chain.post.normalize_peak(final_waveform, target_db=-1.0)
            
        elif normalization_mode == "Peak -3dB":
            logger.info("Applying Normalization: Peak -3.0 dB")
            final_waveform = self.chain.post.normalize_peak(final_waveform, target_db=-3.0)
            
        # Legacy "None" does nothing

        
        # Safety Limiter: Ensure audio never exceeds [-1, 1] to prevent clipping
        # This runs regardless of whether normalization was applied or not
        max_peak = torch.max(torch.abs(final_waveform))
        if max_peak > 1.0:
            logger.warning(f"Audio clipping detected (Peak: {max_peak:.2f}). Clamping to [-1.0, 1.0].")
            final_waveform = torch.clamp(final_waveform, -1.0, 1.0)

        if progress_callback:
            progress_callback(0.9, "Saving output...")
            
        # Move to CPU for saving and analysis
        final_waveform = final_waveform.cpu()
        
        self.save_audio(final_waveform, output_path, self.config.target_sample_rate)
        
        results = {"output_path": output_path}
        
        # 4. Analysis
        if generate_analysis:
            if progress_callback:
                progress_callback(0.95, "Generating spectrograms...")
            from .analysis import generate_spectrogram, generate_psd_plot, generate_waveform_zoom
            
            logger.info("Generating analysis plots...")
            
            # Spectrograms
            results["input_spectrogram"] = generate_spectrogram(input_waveform, input_sr, "Input Spectrogram")
            results["output_spectrogram"] = generate_spectrogram(final_waveform, self.config.target_sample_rate, "Output Spectrogram")
            
            # PSD Plots (Frequency Spectrum)
            results["input_psd"] = generate_psd_plot(input_waveform, input_sr, "Input Frequency Spectrum")
            results["output_psd"] = generate_psd_plot(final_waveform, self.config.target_sample_rate, "Output Frequency Spectrum")
            
            # Waveform Zoom (Transient Detail)
            results["input_wave"] = generate_waveform_zoom(input_waveform, input_sr, title="Input Waveform Detail")
            results["output_wave"] = generate_waveform_zoom(final_waveform, self.config.target_sample_rate, title="Output Waveform Detail")
            
        if progress_callback:
            progress_callback(1.0, "Done!")
        logger.info("Processing complete.")
        return results
