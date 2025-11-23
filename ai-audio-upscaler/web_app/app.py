import gradio as gr
import os
import sys
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AUTO-FIX: Add FFmpeg to PATH if installed via Winget but not yet visible
# This MUST happen before any other imports that might use torchaudio/ffmpeg
try:
    import glob
    import shutil
    
    # 1. Check Winget Links (Preferred)
    winget_links = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Links")
    if os.path.exists(os.path.join(winget_links, "ffmpeg.exe")) and winget_links not in os.environ["PATH"]:
        logger.info(f"Found FFmpeg in Winget Links, adding to PATH: {winget_links}")
        os.environ["PATH"] += os.pathsep + winget_links
        
    # 2. Check Winget Packages (Fallback)
    else:
        winget_packages = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages")
        ffmpeg_dirs = glob.glob(os.path.join(winget_packages, "Gyan.FFmpeg_*", "ffmpeg-*", "bin"))
        
        if ffmpeg_dirs:
            winget_ffmpeg_path = ffmpeg_dirs[0]
            if winget_ffmpeg_path not in os.environ["PATH"]:
                logger.info(f"Found FFmpeg in Packages, adding to PATH: {winget_ffmpeg_path}")
                os.environ["PATH"] += os.pathsep + winget_ffmpeg_path

    # Verify
    if shutil.which("ffmpeg"):
        logger.info(f"FFmpeg detected at: {shutil.which('ffmpeg')}")
    else:
        logger.warning("FFmpeg still not found in PATH.")

except Exception as e:
    logger.error(f"Error trying to auto-add FFmpeg to PATH: {e}")

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from ai_audio_upscaler.ai_upscaler.metrics import calculate_lsd, calculate_ssim
import torchaudio

# Log available backends for debugging
logger.info(f"Torchaudio Backends: {torchaudio.list_audio_backends()}")

from ai_audio_upscaler.config import UpscalerConfig
from ai_audio_upscaler.pipeline import AudioUpscalerPipeline
from ai_audio_upscaler.ai_upscaler.inference import AIUpscalerWrapper
from train import train_model, find_audio_files

def scan_dataset_ui(data_dir):
    if not data_dir or not os.path.exists(data_dir):
        return "Error: Invalid Directory"
    
    files = find_audio_files(data_dir)
    if not files:
        return "No audio files found."
    
    # Format as a simple text list
    file_list = "\n".join(files)
    return f"Found {len(files)} files:\n\n{file_list}"

def run_training_ui(data_dir, epochs, batch_size, lr, save_name, device, use_gan, progress=gr.Progress()):
    if not data_dir or not os.path.exists(data_dir):
        return "Error: Invalid Data Directory", None
        
    save_path = os.path.join("checkpoints", f"{save_name}.ckpt")
    
    # Lists to store loss for plotting
    epoch_list = []
    loss_list = []
    
    # Callback for loss plotting
    def loss_callback(epoch, loss):
        epoch_list.append(epoch)
        loss_list.append(loss)
        
    # Wrapper for progress
    def progress_wrapper(p, msg):
        progress(p, desc=msg)

    try:
        msg = train_model(
            data_dir=data_dir,
            save_path=save_path,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            device=device.lower(),
            use_gan=use_gan,
            progress_callback=progress_wrapper,
            yield_loss=loss_callback
        )
        
        # Generate final plot
        fig, ax = plt.subplots()
        ax.plot(epoch_list, loss_list, marker='o')
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L1 Loss")
        ax.grid(True)
        
        return msg, fig
        
    except Exception as e:
        return f"Error: {str(e)}", None

def run_benchmark_ui(bench_dir, model_name, device, progress=gr.Progress()):
    if not bench_dir or not os.path.exists(bench_dir):
        return "Error: Invalid Directory", None
        
    files = find_audio_files(bench_dir)
    if not files:
        return "No audio files found.", None
        
    # Resolve model path
    model_checkpoint = None
    if model_name and model_name != "None":
        model_checkpoint = os.path.join("checkpoints", model_name)
        
    # Config for upscaling
    # We assume 48kHz target for benchmarking standard
    config = UpscalerConfig(
        target_sample_rate=48000,
        mode="ai" if model_checkpoint else "baseline",
        baseline_method="sinc",
        model_checkpoint=model_checkpoint,
        device=device.lower(),
        export_format="wav"
    )
    
    pipeline = AudioUpscalerPipeline(config)
    
    lsd_scores = []
    ssim_scores = []
    log = []
    
    # Create temp dir for outputs
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, file_path in enumerate(files):
            filename = os.path.basename(file_path)
            msg = f"Benchmarking {i+1}/{len(files)}: {filename}"
            progress((i / len(files)), desc=msg)
            
            try:
                # 1. Load GT
                gt_waveform, gt_sr = torchaudio.load(file_path)
                
                # Ensure GT is 48kHz for fair comparison
                if gt_sr != 48000:
                    resampler = torchaudio.transforms.Resample(gt_sr, 48000)
                    gt_waveform = resampler(gt_waveform)
                
                # 2. Downsample to create synthetic input (e.g. 24kHz)
                downsampler = torchaudio.transforms.Resample(48000, 24000)
                input_waveform = downsampler(gt_waveform)
                
                # Save temp input
                temp_input = os.path.join(temp_dir, "temp_input.wav")
                torchaudio.save(temp_input, input_waveform, 24000)
                
                # 3. Upscale
                temp_output = os.path.join(temp_dir, "temp_output.wav")
                pipeline.run(temp_input, temp_output, normalize=False, generate_analysis=False)
                
                # 4. Load Upscaled
                upscaled_waveform, _ = torchaudio.load(temp_output)
                
                # Align lengths
                min_len = min(gt_waveform.shape[1], upscaled_waveform.shape[1])
                gt_trimmed = gt_waveform[:, :min_len]
                upscaled_trimmed = upscaled_waveform[:, :min_len]
                
                # 5. Calculate Metrics
                # Compute Spectrograms
                n_fft = 2048
                win_length = 1200
                hop_length = 300
                window = torch.hann_window(win_length).to(gt_trimmed.device)
                
                gt_spec = torch.stft(gt_trimmed, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                up_spec = torch.stft(upscaled_trimmed, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
                
                gt_mag = torch.abs(gt_spec)
                up_mag = torch.abs(up_spec)
                
                lsd = calculate_lsd(gt_mag, up_mag)
                ssim = calculate_ssim(gt_mag, up_mag)
                
                lsd_scores.append(lsd)
                ssim_scores.append(ssim)
                
                log.append(f"{filename}: LSD={lsd:.4f}, SSIM={ssim:.4f}")
                
            except Exception as e:
                log.append(f"Error {filename}: {e}")
                
    # Aggregate Results
    avg_lsd = sum(lsd_scores) / len(lsd_scores) if lsd_scores else 0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    
    summary = f"=== Benchmark Results ===\nFiles Tested: {len(lsd_scores)}\nAverage LSD: {avg_lsd:.4f} (Lower is better)\nAverage SSIM: {avg_ssim:.4f} (Higher is better)\n\nDetails:\n" + "\n".join(log)
    
    # Plot
    fig, ax = plt.subplots()
    ax.hist(lsd_scores, bins=10, alpha=0.7, color='blue')
    ax.set_title("LSD Score Distribution")
    ax.set_xlabel("LSD (Log-Spectral Distance)")
    ax.set_ylabel("Count")
    ax.grid(True)
    
    return summary, fig

def get_available_models():
    """Returns list of available model checkpoints."""
    models = AIUpscalerWrapper.list_available_models()
    return [None] + models # Add None for random/default

def process_audio(input_file, target_rate, mode, baseline_method, model_name, export_format, output_dir, device, normalize, progress=gr.Progress()):
    if input_file is None:
        return None, None, None
    
    # Handle single file or list (Batch processing calls this in loop, but here we handle single)
    # If input_file is a list (from batch), we shouldn't be here directly unless we change logic.
    # For now, let's assume single file for the main tab.
    
    # Determine output path
    if not output_dir or not os.path.isdir(output_dir):
        output_dir = os.path.dirname(input_file) # Default to input directory if invalid/empty
    
    filename = os.path.basename(input_file)
    name, _ = os.path.splitext(filename)
    
    # Construct output filename with correct extension
    ext = export_format.lower()
    output_path = os.path.join(output_dir, f"{name}_upscaled_{int(target_rate)}hz.{ext}")
    
    # Resolve model path
    model_checkpoint = None
    if model_name and model_name != "None":
        model_checkpoint = os.path.join("checkpoints", model_name)

    config = UpscalerConfig(
        target_sample_rate=int(target_rate),
        mode=mode.lower(),
        baseline_method=baseline_method.lower(),
        model_checkpoint=model_checkpoint, 
        device=device.lower(),
        export_format=ext
    )
    
    # Progress callback wrapper for Gradio
    def update_progress(p, msg):
        progress(p, desc=msg)
    
    pipeline = AudioUpscalerPipeline(config)
    results = pipeline.run(
        input_file, 
        output_path, 
        normalize=normalize, 
        generate_analysis=True, 
        progress_callback=update_progress
    )
    
    return results["output_path"], results.get("input_spectrogram"), results.get("output_spectrogram")

def process_batch(files, target_rate, mode, baseline_method, model_name, export_format, output_dir, device, normalize, progress=gr.Progress()):
    if not files:
        return "No files selected."
        
    if not output_dir:
        return "Error: Output directory required for batch processing."
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    results_log = []
    
    # Resolve model path once
    model_checkpoint = None
    if model_name and model_name != "None":
        model_checkpoint = os.path.join("models", model_name)

    config = UpscalerConfig(
        target_sample_rate=int(target_rate),
        mode=mode.lower(),
        baseline_method=baseline_method.lower(),
        model_checkpoint=model_checkpoint, 
        device=device.lower(),
        export_format=export_format.lower()
    )
    
    pipeline = AudioUpscalerPipeline(config)
    
    for i, input_path in enumerate(files):
        # files is now a list of strings (paths)
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        ext = export_format.lower()
        output_path = os.path.join(output_dir, f"{name}_upscaled.{ext}")
        
        msg = f"Processing {i+1}/{len(files)}: {filename}"
        progress((i / len(files)), desc=msg)
        
        try:
            pipeline.run(input_path, output_path, normalize=normalize, generate_analysis=False)
            results_log.append(f"✅ {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            results_log.append(f"❌ {filename}: {str(e)}")
            
    return "\n".join(results_log)

def main():
    with gr.Blocks(title="AI Audio Up-Scaler") as demo:
        gr.Markdown("# 🎵 AI Audio Up-Scaler Pro")
        
        with gr.Tabs():
            # --- Tab 1: Single File Upscale ---
            with gr.TabItem("Upscale (Single)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Settings")
                        input_audio = gr.Audio(type="filepath", label="Input Audio")
                        
                        with gr.Group():
                            target_rate = gr.Number(value=48000, label="Target Sample Rate (Hz)")
                            export_format = gr.Dropdown(["WAV", "FLAC", "MP3", "OGG"], value="WAV", label="Export Format")
                        
                        with gr.Group():
                            mode = gr.Radio(["Baseline", "AI"], value="Baseline", label="Upscaling Mode")
                            baseline_method = gr.Radio(["Sinc", "Linear"], value="Sinc", label="Baseline Method (if Mode=Baseline)")
                            
                            # Dynamic model list
                            model_choices = get_available_models()
                            model_name = gr.Dropdown(model_choices, value=None, label="AI Model Checkpoint (if Mode=AI)")
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            device_choices = ["CPU"]
                            if torch.cuda.is_available():
                                device_choices.insert(0, "CUDA")
                            device = gr.Radio(device_choices, value=device_choices[0], label="Device")
                            output_dir = gr.Textbox(label="Output Directory", placeholder="Leave empty for same as input")
                            normalize = gr.Checkbox(label="Normalize Output (-1 dB)", value=True)
                        
                        submit_btn = gr.Button("🚀 Upscale Audio", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Results & Analysis")
                        output_audio = gr.Audio(label="Upscaled Output", type="filepath", interactive=False)
                        
                        with gr.Tab("Spectrograms"):
                            input_spec = gr.Plot(label="Input")
                            output_spec = gr.Plot(label="Output")
                
                submit_btn.click(
                    fn=process_audio,
                    inputs=[input_audio, target_rate, mode, baseline_method, model_name, export_format, output_dir, device, normalize],
                    outputs=[output_audio, input_spec, output_spec]
                )

            # --- Tab 2: Batch Processing ---
            with gr.TabItem("Batch Processing"):
                gr.Markdown("### Bulk Upscaling")
                gr.Markdown("Upload multiple files to process them all at once.")
                
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(file_count="multiple", label="Input Files", type="filepath")
                        batch_output_dir = gr.Textbox(label="Output Directory (Required)", placeholder="C:/Users/Name/Music/Upscaled")
                        
                        # Re-use settings logic or duplicate controls? 
                        # Duplicating for clarity in batch context
                        with gr.Group():
                            b_target_rate = gr.Number(value=48000, label="Target Rate")
                            b_format = gr.Dropdown(["WAV", "FLAC", "MP3", "OGG"], value="WAV", label="Format")
                            b_mode = gr.Radio(["Baseline", "AI"], value="Baseline", label="Mode")
                            b_model = gr.Dropdown(get_available_models(), value=None, label="Model")
                            b_device = gr.Radio(device_choices, value=device_choices[0], label="Device")
                            b_norm = gr.Checkbox(label="Normalize", value=True)
                            
                        batch_btn = gr.Button("Start Batch Processing", variant="primary")
                        
                        # Hidden state for baseline method (default to Sinc for batch)
                        b_baseline_method = gr.State(value="Sinc")
                        
                    with gr.Column():
                        batch_log = gr.Textbox(label="Processing Log", lines=20)
                
                batch_btn.click(
                    fn=process_batch,
                    inputs=[batch_files, b_target_rate, b_mode, b_baseline_method, b_model, b_format, batch_output_dir, b_device, b_norm],
                    outputs=[batch_log]
                )

            # --- Tab 2: Train ---
            with gr.TabItem("Train Model"):
                gr.Markdown("Train a custom AI model on your own high-quality audio files.")
                gr.Markdown("1. Put your high-quality WAV/FLAC files in a folder (subfolders are searched recursively).\n2. Enter the folder path below.\n3. Click 'Scan Dataset' to verify files.\n4. Click 'Start Training'.")
                
                with gr.Row():
                    with gr.Column():
                        data_dir = gr.Textbox(label="Dataset Directory (Absolute Path)", placeholder="C:/Users/Name/Music/HighRes")
                        scan_btn = gr.Button("Scan Dataset")
                        file_list_box = gr.Textbox(label="Found Files", lines=5, max_lines=10)
                        
                        save_name = gr.Textbox(label="Model Name", value="my_custom_model")
                        
                        with gr.Row():
                            epochs = gr.Number(value=5, label="Epochs", precision=0)
                            batch_size = gr.Number(value=4, label="Batch Size", precision=0)
                            lr = gr.Number(value=0.0001, label="Learning Rate")
                        
                        train_device_choices = ["CPU"]
                        if torch.cuda.is_available():
                            train_device_choices.insert(0, "CUDA")
                        train_device = gr.Radio(train_device_choices, value=train_device_choices[0], label="Training Device")
                        use_gan = gr.Checkbox(label="Enable GAN Training (Slower, Better Quality)", value=False)
                        
                        train_btn = gr.Button("Start Training", variant="primary")
                    
                    with gr.Column():
                        train_status = gr.Textbox(label="Status")
                        loss_plot = gr.Plot(label="Training Loss Curve")
                
                scan_btn.click(
                    fn=scan_dataset_ui,
                    inputs=[data_dir],
                    outputs=[file_list_box]
                )
                
                train_btn.click(
                    fn=run_training_ui,
                    inputs=[data_dir, epochs, batch_size, lr, save_name, train_device, use_gan],
                    outputs=[train_status, loss_plot]
                )

            # --- Tab 4: Benchmark ---
            with gr.TabItem("Benchmark Model"):
                gr.Markdown("### Model Validation")
                gr.Markdown("Measure the true quality improvement by comparing upscaled audio against original High-Res files.")
                gr.Markdown("1. Select a folder of **High-Quality** (Ground Truth) audio.\n2. The system will downsample it, upscale it with your model, and compare the results.")
                
                with gr.Row():
                    with gr.Column():
                        bench_dir = gr.Textbox(label="Ground Truth Directory (Absolute Path)", placeholder="C:/Users/Name/Music/HighRes_Test")
                        
                        # Benchmark Settings
                        with gr.Group():
                            bench_model = gr.Dropdown(get_available_models(), value=None, label="Model to Test")
                            bench_device = gr.Radio(device_choices, value=device_choices[0], label="Device")
                        
                        bench_btn = gr.Button("Run Benchmark", variant="primary")
                        
                    with gr.Column():
                        bench_results = gr.Textbox(label="Benchmark Results", lines=10)
                        bench_plot = gr.Plot(label="LSD Score Distribution (Lower is Better)")
                
                bench_btn.click(
                    fn=run_benchmark_ui,
                    inputs=[bench_dir, bench_model, bench_device],
                    outputs=[bench_results, bench_plot]
                )
        
    demo.launch()

if __name__ == "__main__":
    main()
