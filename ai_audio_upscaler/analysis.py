import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torchaudio

def generate_spectrogram(waveform: torch.Tensor, sample_rate: int, title: str = "Spectrogram") -> plt.Figure:
    """
    Generates a spectrogram image from an audio waveform.
    
    Args:
        waveform (torch.Tensor): Audio data (Channels, Time).
        sample_rate (int): Sample rate in Hz.
        title (str): Title for the plot.
        
    Returns:
        plt.Figure: Matplotlib figure object containing the plot.
    """
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Compute STFT
    n_fft = 2048
    win_length = None
    hop_length = 512
    
    spectrogram = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=torch.hann_window(n_fft).to(waveform.device),
        return_complex=True
    )
    
    # Convert to magnitude and log scale
    magnitude = torch.abs(spectrogram)
    log_spectrogram = 20 * torch.log10(magnitude + 1e-9)
    
    # Plot with dark theme
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        # Make figure background transparent to blend with UI
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        img = ax.imshow(
            log_spectrogram[0].cpu().numpy(), 
            aspect='auto', 
            origin='lower', 
            cmap='inferno', # Better contrast for dark mode
            extent=[0, waveform.shape[1] / sample_rate, 0, sample_rate / 2]
        )
        ax.set_title(title, color='white', fontweight='bold')
        ax.set_ylabel("Frequency (Hz)", color='#cccccc')
        ax.set_xlabel("Time (s)", color='#cccccc')
        
        # Style ticks
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#cccccc')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')
        
        plt.tight_layout()
    
    return fig

def generate_psd_plot(waveform: torch.Tensor, sample_rate: int, title: str = "Power Spectral Density"):
    """
    Generates a Power Spectral Density (PSD) plot.
    """
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    audio_np = waveform.squeeze().cpu().numpy()
    
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # Compute PSD using Welch's method
        ax.psd(audio_np, NFFT=4096, Fs=sample_rate, color='#00ffff', alpha=0.9, linewidth=1.5)
        
        ax.set_title(title, color='white', fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", color='#cccccc')
        ax.set_ylabel("Power Spectral Density (dB/Hz)", color='#cccccc')
        ax.grid(True, which='both', alpha=0.2, color='#444444')
        
        # Highlight Nyquist
        ax.axvline(x=22050, color='#ffaa00', linestyle='--', alpha=0.8, label='CD Limit (22kHz)')
        
        # Style legend
        legend = ax.legend(facecolor='#222222', edgecolor='#444444')
        plt.setp(legend.get_texts(), color='#cccccc')
        
        # Style ticks
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        
        plt.tight_layout()
    return fig

def generate_waveform_zoom(waveform: torch.Tensor, sample_rate: int, duration_ms: float = 10.0, title: str = "Waveform Detail"):
    """
    Generates a zoomed-in waveform plot to show transient detail.
    """
    # Take a slice from the middle
    mid_point = waveform.shape[1] // 2
    samples_to_show = int((duration_ms / 1000.0) * sample_rate)
    start = mid_point
    end = start + samples_to_show
    
    if end > waveform.shape[1]:
        start = 0
        end = samples_to_show
        
    slice_np = waveform[0, start:end].cpu().numpy()
    time_axis = np.linspace(0, duration_ms, len(slice_np))
    
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        ax.plot(time_axis, slice_np, linewidth=1.5, color='#00ffff')
        ax.set_title(f"{title} ({duration_ms}ms slice)", color='white', fontweight='bold')
        ax.set_xlabel("Time (ms)", color='#cccccc')
        ax.set_ylabel("Amplitude", color='#cccccc')
        ax.grid(True, alpha=0.2, color='#444444')
        
        # Style ticks
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        
        plt.tight_layout()
    return fig

def generate_delta_spectrogram(input_waveform: torch.Tensor, output_waveform: torch.Tensor, sample_rate: int, title: str = "Δ Difference Spectrogram"):
    """
    Generates a 'ghost' spectrogram showing only the added/removed content.
    """
    # Ensure dimensions match
    if input_waveform.shape[-1] != output_waveform.shape[-1]:
        # Resample input to match output length if needed (simple linear for viz)
        input_waveform = torch.nn.functional.interpolate(input_waveform.unsqueeze(0), size=output_waveform.shape[-1], mode='linear').squeeze(0)

    # Convert to mono
    if input_waveform.shape[0] > 1: input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
    if output_waveform.shape[0] > 1: output_waveform = torch.mean(output_waveform, dim=0, keepdim=True)

    # Compute STFTs
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(input_waveform.device)
    
    stft_in = torch.stft(input_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    stft_out = torch.stft(output_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    # Compute Magnitude Difference
    mag_in = torch.abs(stft_in)
    mag_out = torch.abs(stft_out)
    
    # Difference: What did the model add? (Output - Input)
    # We use a diverging colormap: Red = Added, Blue = Removed
    diff = 20 * torch.log10(mag_out + 1e-9) - 20 * torch.log10(mag_in + 1e-9)
    
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        img = ax.imshow(
            diff[0].cpu().numpy(), 
            aspect='auto', 
            origin='lower', 
            cmap='seismic', # Diverging: Blue (Negative) <-> White (Zero) <-> Red (Positive)
            vmin=-20, vmax=20, # Clamp range for visibility
            extent=[0, output_waveform.shape[1] / sample_rate, 0, sample_rate / 2]
        )
        ax.set_title(title, color='white', fontweight='bold')
        ax.set_ylabel("Frequency (Hz)", color='#cccccc')
        ax.set_xlabel("Time (s)", color='#cccccc')
        
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#cccccc')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')
        
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values(): spine.set_edgecolor('#444444')
        plt.tight_layout()
        
    return fig

def generate_vectorscope(waveform: torch.Tensor, title: str = "Stereo Vectorscope"):
    """
    Generates a Goniometer (Lissajous) plot for stereo analysis.
    """
    if waveform.shape[0] < 2:
        return None # Mono has no vectorscope
        
    # Downsample for performance if too long
    max_samples = 100000
    if waveform.shape[1] > max_samples:
        step = waveform.shape[1] // max_samples
        left = waveform[0, ::step].cpu().numpy()
        right = waveform[1, ::step].cpu().numpy()
    else:
        left = waveform[0].cpu().numpy()
        right = waveform[1].cpu().numpy()
        
    # Rotate 45 degrees to get Mid/Side axes
    # Mid = L+R, Side = L-R
    # Standard Goniometer: X = Side, Y = Mid
    mid = left + right
    side = left - right
    
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # 2D Histogram for density
        ax.hist2d(side, mid, bins=100, cmap='inferno', range=[[-1.5, 1.5], [-1.5, 1.5]])
        
        ax.set_title(title, color='white', fontweight='bold')
        ax.set_xlabel("Side (L-R)", color='#cccccc')
        ax.set_ylabel("Mid (L+R)", color='#cccccc')
        ax.grid(True, alpha=0.2, color='#444444', linestyle='--')
        
        # Draw axes lines
        ax.axhline(0, color='white', alpha=0.3)
        ax.axvline(0, color='white', alpha=0.3)
        
        # Diagonal lines (L/R axes)
        x = np.linspace(-1.5, 1.5, 10)
        ax.plot(x, x, color='cyan', alpha=0.2, label='Left')
        ax.plot(x, -x, color='orange', alpha=0.2, label='Right')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values(): spine.set_edgecolor('#444444')
        plt.tight_layout()
        
    return fig

def get_audio_metadata(path: str):
    """
    Extracts technical metadata from an audio file.
    """
    if not os.path.exists(path):
        return {"Error": "File not found"}
        
    try:
        info = torchaudio.info(path)
        
        # Calculate duration
        duration_sec = info.num_frames / info.sample_rate
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        
        # Estimate bitrate (not always available in info, approximate)
        file_size = os.path.getsize(path)
        bitrate_kbps = int((file_size * 8) / duration_sec / 1000)
        
        return {
            "Format": os.path.splitext(path)[1].upper().replace(".", ""),
            "Sample Rate": f"{info.sample_rate} Hz",
            "Channels": f"{info.num_channels} ({'Stereo' if info.num_channels == 2 else 'Mono'})",
            "Bit Depth": "32-bit Float" if info.bits_per_sample == 0 else f"{info.bits_per_sample}-bit", # Torch often reports 0 for float
            "Duration": f"{mins}:{secs:02d}",
            "Bitrate": f"~{bitrate_kbps} kbps"
        }
    except Exception as e:
        return {"Error": str(e)}

def calculate_loudness_stats(waveform: torch.Tensor, sample_rate: int):
    """
    Calculates Integrated Loudness (LUFS) and True Peak (dBTP).
    """
    import pyloudnorm as pyln
    
    try:
        meter = pyln.Meter(sample_rate)
        # Pyloudnorm expects (samples, channels)
        audio_np = waveform.cpu().t().numpy()
        
        lufs = meter.integrated_loudness(audio_np)
        
        # True Peak (approximate with max abs)
        peak = torch.max(torch.abs(waveform)).item()
        peak_db = 20 * np.log10(peak + 1e-9)
        
        return {
            "LUFS": f"{lufs:.1f}",
            "True Peak": f"{peak_db:.1f} dBTP"
        }
    except Exception:
        return {"LUFS": "-inf", "True Peak": "-inf"}

# --- New Analysis Tools ---

def detect_cutoff_frequency(waveform: torch.Tensor, sample_rate: int, sensitivity: str = "adaptive"):
    """
    Detects the frequency where energy drops below a threshold (Fake High-Res Detector).
    Returns the estimated cutoff frequency in Hz.
    
    Args:
        sensitivity: "adaptive" (uses noise floor) or "strict" (fixed threshold).
    """
    # Convert to mono and ensure 1D
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0)
        
    # Compute PSD using Welch's method (using matplotlib's mlab for convenience or torch)
    # We'll use torch.stft and average over time
    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft).to(waveform.device)
    
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft)
    
    # Average magnitude over time -> Frequency Spectrum
    # magnitude is (Freq, Time) since input is 1D
    avg_mag = torch.mean(magnitude, dim=1)
    avg_mag_db = 20 * torch.log10(avg_mag + 1e-9)
    
    # Normalize to 0dB max
    peak_db = torch.max(avg_mag_db)
    avg_mag_db = avg_mag_db - peak_db
    
    # Determine Threshold
    if sensitivity == "adaptive":
        # Estimate Noise Floor from the top 10% of frequencies (usually quietest)
        noise_floor_db = torch.quantile(avg_mag_db, 0.1) # Bottom 10%
        # Threshold is 6dB above noise floor, but capped at -90dB max
        threshold_db = max(noise_floor_db + 6.0, -90.0)
    else:
        threshold_db = -60.0
    
    # Frequency bins
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    
    # Smooth the spectrum
    kernel_size = 3
    avg_mag_db_smooth = torch.nn.functional.avg_pool1d(avg_mag_db.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2).squeeze()
    
    # Scan BACKWARDS from Nyquist to find the first frequency ABOVE threshold
    cutoff_idx = -1
    start_idx = int((15000 / (sample_rate / 2)) * len(freqs))
    
    # Skip the very last few bins (Nyquist edge) to avoid artifacts
    end_idx = len(avg_mag_db_smooth) - 10
    
    for i in range(end_idx, start_idx, -1):
        if avg_mag_db_smooth[i] > threshold_db:
            # Check for robustness: Ensure 3 consecutive bins are above threshold
            # to avoid triggering on single noise spikes
            if (i - 2 >= 0) and \
               (avg_mag_db_smooth[i-1] > threshold_db) and \
               (avg_mag_db_smooth[i-2] > threshold_db):
                cutoff_idx = i
                break
            
    if cutoff_idx != -1:
        return freqs[cutoff_idx].item()
    else:
        return 15000.0

def calculate_lra(waveform: torch.Tensor, sample_rate: int):
    """
    Calculates Loudness Range (LRA) - a measure of dynamic range.
    Uses pyloudnorm if available.
    """
    import pyloudnorm as pyln
    try:
        meter = pyln.Meter(sample_rate)
        audio_np = waveform.cpu().t().numpy()
        # Note: pyloudnorm doesn't have a direct LRA method in the simple API, 
        # but we can approximate or check if it's available.
        # Actually, standard pyloudnorm usually only does Integrated.
        # Let's implement a simplified LRA: 
        # LRA = difference between 95th and 10th percentile of short-term loudness.
        
        # Calculate short-term loudness (3s window)
        # This is complex to implement from scratch. 
        # Let's return a placeholder or use RMS variance.
        
        # RMS Variance approach (Simple Dynamic Range)
        frame_size = int(0.1 * sample_rate) # 100ms frames
        unfolded = waveform.unfold(1, frame_size, frame_size)
        rms = torch.sqrt(torch.mean(unfolded**2, dim=2))
        rms_db = 20 * torch.log10(rms + 1e-9)
        
        # 95th - 10th percentile
        q95 = torch.quantile(rms_db, 0.95)
        q10 = torch.quantile(rms_db, 0.10)
        lra = q95 - q10
        
        return f"{lra.item():.1f} LU"
        
    except Exception:
        return "N/A"

def estimate_noise_floor(waveform: torch.Tensor):
    """
    Estimates the noise floor by looking at the quietest parts of the audio.
    """
    # Frame-based RMS
    frame_size = 2048
    if waveform.shape[1] < frame_size:
        return "-inf dB"
        
    unfolded = waveform.unfold(1, frame_size, frame_size)
    rms = torch.sqrt(torch.mean(unfolded**2, dim=2))
    rms_db = 20 * torch.log10(rms + 1e-9)
    
    # Use the 1st percentile (bottom 1%) as the noise floor
    noise_floor = torch.quantile(rms_db, 0.01)
    
    return f"{noise_floor.item():.1f} dB"

def count_clipping(waveform: torch.Tensor):
    """
    Counts samples that hit exactly 1.0 or -1.0 (or close to it).
    """
    threshold = 0.999
    clipped_samples = torch.sum(torch.abs(waveform) >= threshold).item()
    return clipped_samples

from ai_audio_upscaler.audio_io import load_audio_robust

def analyze_file_quality(file_path: str):
    """
    Runs a full suite of quality checks on a file.
    """
    try:
        waveform, sr = load_audio_robust(file_path)
        
        # 1. Cutoff Detection
        cutoff = detect_cutoff_frequency(waveform, sr)
        
        # 2. Dynamics
        lra = calculate_lra(waveform, sr)
        
        # 3. Noise
        noise = estimate_noise_floor(waveform)
        
        # 4. Clipping
        clips = count_clipping(waveform)
        
        # 5. Metadata
        meta = get_audio_metadata(file_path)
        
        # Verdict Logic
        status = "✅ PASS"
        issues = []
        
        # Check Cutoff (Fake High-Res)
        # If SR > 48k but cutoff < 22k -> Fake
        # Check Cutoff (Fake High-Res)
        # If SR > 48k check for bandwidth
        if sr > 48000:
            if cutoff < 18000:
                status = "❌ FAIL"
                issues.append(f"Fake High-Res (Cutoff @ {cutoff/1000:.1f}kHz)")
            elif cutoff < 21000:
                status = "⚠️ WARN"
                issues.append(f"Low Bandwidth (Cutoff @ {cutoff/1000:.1f}kHz)")
            # Else PASS (Cutoff >= 21k is acceptable for 96k files if source was analog/mic limited)
             
        # Check Clipping
        if clips > 100:
            issues.append(f"Heavy Clipping ({clips} samples)")
            if status == "✅ PASS": status = "⚠️ WARN"
            
        # Check Bit Depth (if 16-bit in 96k container)
        # (Hard to detect from float tensor, relying on metadata if possible)
        
        return {
            "Filename": os.path.basename(file_path),
            "Status": status,
            "Cutoff": f"{cutoff/1000:.1f} kHz",
            "LRA": lra,
            "Noise Floor": noise,
            "Clipping": clips,
            "Issues": ", ".join(issues) if issues else "None",
            "Sample Rate": meta.get("Sample Rate", "N/A"),
            "Bit Depth": meta.get("Bit Depth", "N/A")
        }
        
    except Exception as e:
        return {
            "Filename": os.path.basename(file_path),
            "Status": "❌ ERROR",
            "Issues": str(e)
        }
