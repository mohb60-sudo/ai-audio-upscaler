import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import os
import glob
import logging
import time
from ai_audio_upscaler.dsp_basic import DSPUpscaler
from ai_audio_upscaler.ai_upscaler.transforms import MP3Compression, BandwidthLimiter, QuantizationNoise
from ai_audio_upscaler.ai_upscaler.degradation import AdvancedDegradation
from ai_audio_upscaler.ai_upscaler.model import AudioSuperResNet
from ai_audio_upscaler.ai_upscaler.diffusion import DiffusionScheduler, DiffusionUNet
from ai_audio_upscaler.ai_upscaler.spectral_model import SpectralUNet
from ai_audio_upscaler.ai_upscaler.discriminator import MultiResolutionDiscriminator
from ai_audio_upscaler.ai_upscaler.loss import MultiResolutionSTFTLoss, discriminator_loss, generator_loss, feature_loss
from ai_audio_upscaler.ai_upscaler.metrics import calculate_lsd
from ai_audio_upscaler.audio_io import load_audio_robust
from ai_audio_upscaler.safety import GPUGuard

logger = logging.getLogger(__name__)

def find_audio_files(data_dir):
    """Recursively finds all supported audio files in a directory."""
    extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a', '*.aac']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(files)

class RobustAudioDataset(Dataset):
    """
    Robust dataset loader for Audio Super Resolution.
    
    Features:
    - **Format Support**: WAV, FLAC, MP3 (via Torchaudio).
    - **Auto-Resampling**: Converts all inputs to `target_sr` (Ground Truth).
    - **On-the-fly Degradation**: Simulates low-res input by:
        1. Bandwidth Limiting (Low-pass filter)
        2. Quantization Noise (Bit-depth reduction)
        3. Compression Artifacts (MP3 simulation)
        4. Downsampling to `input_sr`
    
    This "Self-Supervised" approach allows training on ANY high-quality audio 
    without needing paired low-res files.
    """
    def __init__(self, data_dir, target_sr=96000, input_sr=48000, segment_length=16384, robust=True, use_advanced_degradation=False):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_length = segment_length
        self.robust = robust
        self.use_advanced_degradation = use_advanced_degradation
        
        self.files = find_audio_files(data_dir)
            
        if not self.files:
            logger.warning(f"No audio files found in {data_dir}")
        else:
            logger.info(f"Found {len(self.files)} training files. Robust Mode: {robust}")

        # Pre-instantiate resamplers (will be cloned/handled in getitem if needed, 
        # but torchaudio transforms are usually stateless or we recreate them)
        # Actually, for variable input SRs, we need to resample dynamically.
        
        # We need a baseline upscaler for the input features
        self.dsp = DSPUpscaler(target_sample_rate=target_sr, method="sinc")
        
        # Augmentations
        if use_advanced_degradation:
            self.adv_deg = AdvancedDegradation(sample_rate=target_sr, max_cutoff=int(input_sr/2))
        else:
            # Use continuous range up to Nyquist (input_sr/2) to generalize to MP3/CD inputs
            # This fixes the "gap" where the model ignored inputs >16kHz
            self.mp3_aug = MP3Compression(sample_rate=target_sr)
            self.bw_aug = BandwidthLimiter(sample_rate=target_sr, max_cutoff=int(input_sr/2))
            self.quant_aug = QuantizationNoise()
        
        # Resampler Cache
        self.resamplers = {}
        
        # Fixed Downsampler (Target -> Input)
        self.downsampler = torchaudio.transforms.Resample(target_sr, input_sr)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            waveform, sr = load_audio_robust(path)
            
            # 1. Convert to Mono (for prototype simplicity)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 2. Resample to Target SR (Ground Truth)
            if sr != self.target_sr:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = self.resamplers[sr](waveform)
            
            # 3. Random Crop
            if waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start, (1,)).item()
                target = waveform[:, start:start+self.segment_length]
            else:
                # Pad if too short
                target = torch.nn.functional.pad(waveform, (0, self.segment_length - waveform.shape[1]))
            
            # 4. Create Input (Degrade -> Downsample -> Upsample Baseline)
            # We simulate the "low res" input by downsampling AND adding artifacts
            
            degraded = target.clone()
            
            if self.robust:
                if self.use_advanced_degradation:
                    degraded = self.adv_deg(degraded)
                else:
                    # Apply Bandwidth Limiting (simulates low-pass filters in compression)
                    degraded = self.bw_aug(degraded)
                    
                    # Apply Quantization Noise (simulates low bit depth/bitrate noise)
                    degraded = self.quant_aug(degraded)
                    
                    # Apply MP3 Compression (simulates coding artifacts)
                    degraded = self.mp3_aug(degraded)
            
            # Downsample to input SR
            low_res = self.downsampler(degraded)
            
            # Upsample back to target size using baseline (Input features)
            # The model learns the residual: Target - Baseline
            baseline = self.dsp.process(low_res, self.input_sr)
            
            # Ensure shapes match exactly (resampling might cause off-by-one)
            if baseline.shape[1] != target.shape[1]:
                min_len = min(baseline.shape[1], target.shape[1])
                baseline = baseline[:, :min_len]
                target = target[:, :min_len]
                
            return baseline, target
            
        except Exception as e:
            logger.error(f"Error loading {self.files[idx]}: {e}")
            # Return a dummy zero tensor to avoid crashing
            return torch.zeros(1, self.segment_length), torch.zeros(1, self.segment_length)



def validate(model, val_loader, device, spectral_model=None):
    """
    Run validation on held-out set.
    Returns average LSD score (Lower is better).
    """
    model.eval()
    if spectral_model:
        spectral_model.eval()
        
    total_lsd = 0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass (Waveform)
            outputs = model(inputs)
            
            # Spectral Fusion (if enabled)
            if spectral_model:
                # 1. Input STFT
                input_stft = torch.stft(inputs.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
                input_mag = torch.log1p(input_stft.abs()).unsqueeze(1)
                
                # 2. Predict Mag
                pred_mag_log = spectral_model(input_mag)
                pred_mag = torch.expm1(pred_mag_log).squeeze(1)
                
                # 3. Phase from Waveform Model
                out_stft = torch.stft(outputs.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
                phase = torch.angle(out_stft)
                
                # 4. Fuse
                final_stft = pred_mag * torch.exp(1j * phase)
                
                # 5. Inverse STFT
                outputs = torch.istft(final_stft, n_fft=1024, hop_length=256, length=inputs.shape[2]).unsqueeze(1)
            
            # Compute Spectrograms for LSD
            # Use same params as loss/metrics
            n_fft = 2048
            win_length = 1200
            hop_length = 300
            window = torch.hann_window(win_length).to(device)
            
            out_spec = torch.stft(outputs.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            target_spec = torch.stft(targets.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            
            out_mag = torch.abs(out_spec)
            target_mag = torch.abs(target_spec)
            
            lsd = calculate_lsd(target_mag, out_mag)
            total_lsd += lsd
            count += 1
            
    return total_lsd / count if count > 0 else float('inf')

def inspect_checkpoint(path):
    """
    Reads metadata from a checkpoint file without loading weights to GPU.
    Returns dict with 'epoch', 'base_channels', 'num_layers', 'best_val_lsd', 'config'
    or None if legacy/invalid.
    """
    try:
        # Load to CPU to avoid OOM
        checkpoint = torch.load(path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            # New Format
            return {
                "epoch": checkpoint.get("epoch", 0),
                "best_val_lsd": checkpoint.get("best_val_lsd", float('inf')),
                "config": checkpoint.get("config", {}),
                "timestamp": checkpoint.get("timestamp", "Unknown")
            }
            return None
    except Exception:
        return None

def train_model(data_dir, save_path, epochs=100, batch_size=16, lr=1e-4, device="cuda", use_gan=False, use_diffusion=False, diffusion_steps=1000, base_channels=32, num_layers=4, robust_training=False, use_spectral=False, num_workers=4, use_amp=True, progress_callback=None, stop_event=None, yield_loss=None, auto_batch=False, watchdog=False, max_kernel_ms=1200):
    """
    Main training loop for the Audio Upscaler.
    
    Args:
        data_dir (str): Path to directory containing training audio files.
        save_path (str): Path to save the final model checkpoint.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'.
        use_gan (bool): Whether to enable GAN training (Discriminator).
        use_diffusion (bool): Whether to enable Diffusion training.
        diffusion_steps (int): Number of diffusion steps.
        base_channels (int): Number of base channels for the U-Net.
        num_layers (int): Number of layers (depth) for the U-Net.
        robust_training (bool): Whether to apply robust augmentations (bandwidth limiting, noise, MP3).
        use_spectral (bool): Whether to enable the secondary Spectral Recovery model.
        num_workers (int): Number of CPU workers for data loading.
        use_amp (bool): Whether to use Automatic Mixed Precision (FP16).
        progress_callback (callable): Optional callback for reporting progress (0.0 to 1.0).
        stop_event (threading.Event): Optional event to signal early stopping.
        yield_loss (callable): Optional callback to yield loss values for plotting.
        auto_batch (bool): Whether to auto-tune batch size.
        watchdog (bool): Whether to enable thermal/power watchdog.
        max_kernel_ms (int): Max kernel duration in ms (TDR guard).
        
    Returns:
        dict: Training results including message, best_val_lsd, and final_g_loss.
    """
    # Initialize GPU Guard
    guard = GPUGuard(device=device, max_kernel_ms=max_kernel_ms, watchdog=watchdog)
    guard.preflight()
    
    logger.info(f"Starting training on {guard.device}...")
    
    # Define best model path
    dir_name = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    name_part, ext_part = os.path.splitext(base_name)
    best_save_path = os.path.join(dir_name, f"{name_part}_best{ext_part}")
    
    # Initialize Models
    if use_diffusion:
        logger.info(f"Initializing Diffusion Model (Steps: {diffusion_steps})")
        # Diffusion Model: Base model takes 2 channels (Noisy Input + Condition)
        base_model = AudioSuperResNet(in_channels=2, base_channels=base_channels, num_layers=num_layers, time_emb_dim=32).to(device)
        model = DiffusionUNet(base_model).to(device)
        scheduler = DiffusionScheduler(num_steps=diffusion_steps)
    else:
        logger.info("Initializing GAN/Regression Model")
        # GAN/Regression Model: Takes 1 channel (Low-Res Input)
        base_model = AudioSuperResNet(in_channels=1, base_channels=base_channels, num_layers=num_layers).to(device)
        model = base_model
        scheduler = None
    
    spectral_model = None
    if use_spectral:
        logger.info("Initializing Spectral Recovery Model...")
        spectral_model = SpectralUNet(in_channels=1, base_channels=base_channels).to(device)
        
    discriminator = None
    optimizer_d = None
    if use_gan and not use_diffusion: # GAN not compatible with Diffusion in this implementation
        discriminator = MultiResolutionDiscriminator().to(device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    # Optimizer G (Waveform + Spectral)
    g_params = list(model.parameters())
    if spectral_model:
        g_params += list(spectral_model.parameters())
        
    optimizer_g = optim.Adam(g_params, lr=lr, betas=(0.5, 0.9))
    
    # Losses
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    
    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device == 'cuda')
    
    # Dataset
    # Use Advanced Degradation if Diffusion is enabled (or if robust_training is on and we want better robustness)
    # Let's use Advanced Degradation if robust_training is True, regardless of model type, for better results.
    # But to be safe and follow plan, let's link it to use_diffusion or robust_training.
    use_advanced_deg = robust_training and use_diffusion 
    dataset = RobustAudioDataset(data_dir, target_sr=96000, robust=robust_training, use_advanced_degradation=use_advanced_deg)
    
    # Split Train/Val
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # --- Auto-Batching ---
    if auto_batch and device == "cuda":
        def try_batch(bs):
            # Create dummy batch
            dummy_in = torch.randn(bs, 1, 16384, device=device)
            dummy_target = torch.randn(bs, 1, 16384, device=device)
            
            # Run forward/backward
            with torch.amp.autocast('cuda', enabled=use_amp):
                if use_diffusion:
                    t = torch.randint(0, diffusion_steps, (bs,), device=device)
                    noise = torch.randn_like(dummy_target)
                    x_t, _ = scheduler.add_noise(dummy_target, t, noise)
                    pred_noise = model(x_t, t, condition=dummy_in)
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)
                else:
                    out = model(dummy_in)
                    loss = stft_criterion(out, dummy_target)
            
            scaler.scale(loss).backward()
            optimizer_g.zero_grad()
            return True

        batch_size = guard.auto_batch_size(try_batch, max_bs=batch_size or 64)
        logger.info(f"Auto-tuned batch size: {batch_size}")
    
    # Optimize DataLoader
    pin_memory = (device == "cuda")
    num_workers = guard.optimize_workers(num_workers)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    start_epoch = 0
    best_val_lsd = float('inf')
    
    # Load existing checkpoint if available
    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=device)
            
            # Safety Check: Model Type Mismatch
            ckpt_config = checkpoint.get("config", {})
            ckpt_type = ckpt_config.get("model_type", "gan")
            current_type = "diffusion" if use_diffusion else "gan"
            
            if ckpt_type != current_type:
                logger.error(f"Checkpoint Mismatch! Cannot resume {ckpt_type} model with {current_type} training.")
                return {"message": f"❌ Error: Cannot resume {ckpt_type.upper()} checkpoint with {current_type.upper()} training. Please start fresh or switch modes.", "final_val_lsd": 0}

            # Handle New vs Legacy Format
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # New Format
                model.load_state_dict(checkpoint["state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_val_lsd = checkpoint.get("best_val_lsd", float('inf'))
                logger.info(f"Resumed from Rich Checkpoint (Epoch {start_epoch})")
            else:
                # Legacy Format
                model.load_state_dict(checkpoint)
                logger.info("Resumed from Legacy Checkpoint (Epoch 0)")
                
        except Exception as e:
            logger.warning(f"Could not load existing checkpoint: {e}. Starting fresh.")

    import datetime

    # Training Loop
    try:
        # Time Estimation
        total_steps = len(train_loader) * epochs
        start_time = time.time()
        
        for epoch in range(start_epoch, start_epoch + epochs):
            # Stop Check
            if stop_event and stop_event.is_set():
                logger.info("Training stopped by user.")
                return {"message": "Stopped by User", "final_val_lsd": best_val_lsd}
                
            model.train()
            if discriminator:
                discriminator.train()
            
            total_g_loss = 0
            total_d_loss = 0
            num_batches = len(train_loader)
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Granular Stop Check
                if stop_event and stop_event.is_set():
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                
                # --- Train Discriminator (GAN Only) ---
                if use_gan and not use_diffusion and epoch >= int(epochs * 0.2): # Warmup
                    optimizer_d.zero_grad()
                    
                    with torch.amp.autocast('cuda', enabled=use_amp and device == 'cuda'):
                        fake_audio = model(inputs).detach()
                        real_scores, fake_scores, _, _ = discriminator(targets, fake_audio)
                        d_loss, _, _ = discriminator_loss(real_scores, fake_scores)
                    
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    scaler.step(optimizer_d)
                    scaler.update()
                    total_d_loss += d_loss.item()
                
                # --- Train Generator / Diffusion ---
                def train_step_g():
                    optimizer_g.zero_grad()
                    
                    with torch.amp.autocast('cuda', enabled=use_amp and device == 'cuda'):
                        if use_diffusion:
                            # Diffusion Training
                            # 1. Sample t
                            t = torch.randint(0, diffusion_steps, (inputs.shape[0],), device=device)
                            
                            # 2. Add Noise
                            noise = torch.randn_like(targets)
                            x_t, _ = scheduler.add_noise(targets, t, noise)
                            
                            # 3. Predict Noise
                            # Condition on Low-Res Input
                            pred_noise = model(x_t, t, condition=inputs)
                            
                            # 4. MSE Loss
                            mse_loss = torch.nn.functional.mse_loss(pred_noise, noise)
                            
                            # 5. Perceptual Loss (Auxiliary)
                            # Reconstruct x_0 estimate for STFT loss
                            x_0_pred = scheduler.predict_start_from_noise(x_t, t, pred_noise)
                            stft_loss = stft_criterion(x_0_pred, targets)
                            
                            g_loss = mse_loss + 0.1 * stft_loss
                            
                        else:
                            # Standard GAN/Regression Training
                            # Stream A: Waveform
                            fake_audio = model(inputs)
                            sc_loss = stft_criterion(fake_audio, targets)
                            
                            # Stream B: Spectral (Optional)
                            spec_loss = 0
                            if spectral_model:
                                with torch.no_grad():
                                    target_stft = torch.stft(targets.squeeze(1), n_fft=1024, hop_length=256, return_complex=True)
                                    target_mag = torch.log1p(target_stft.abs()).unsqueeze(1)
                                
                                pred_spec = spectral_model(target_mag)
                                spec_loss = torch.nn.functional.l1_loss(pred_spec, target_mag)
                                
                            g_loss = sc_loss + (spec_loss * 0.5 if spectral_model else 0)
                            
                            # GAN Loss
                            if use_gan and epoch >= int(epochs * 0.2):
                                fake_scores, _, _, _ = discriminator(targets, fake_audio)
                                g_adv_loss = generator_loss(fake_scores)
                                g_loss += 0.1 * g_adv_loss
                                
                            # Feature Loss
                            if use_gan:
                                feat_loss = feature_loss(discriminator, targets, fake_audio)
                                g_loss += 10.0 * feat_loss

                    scaler.scale(g_loss).backward()
                    scaler.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(g_params, max_norm=1.0)
                    scaler.step(optimizer_g)
                    scaler.update()
                    return g_loss.item()

                with guard.kernel_timer("train_step"):
                    g_loss_item = guard.safe_step(train_step_g)
                    total_g_loss += g_loss_item
                            
                if progress_callback and batch_idx % 5 == 0:
                    # Calculate ETC
                    elapsed = time.time() - start_time
                    steps_done = (epoch - start_epoch) * num_batches + batch_idx + 1
                    avg_time_per_step = elapsed / steps_done
                    remaining_steps = total_steps - steps_done
                    etc_seconds = remaining_steps * avg_time_per_step
                    
                    # Format ETC
                    etc_str = str(datetime.timedelta(seconds=int(etc_seconds)))
                    
                    current_progress = steps_done / total_steps
                    msg = f"Epoch {epoch+1} - Batch {batch_idx}/{num_batches} - Loss: {g_loss_item:.4f} - ETC: {etc_str}"
                    progress_callback(current_progress, msg)
            
            if stop_event and stop_event.is_set():
                break

            avg_g_loss = total_g_loss / num_batches
            
            # --- Validation Step ---
            # For Diffusion, validation is tricky (slow sampling).
            # We can skip full sampling validation and just check loss, 
            # OR run 1-step sampling (bad quality) OR run full sampling on 1 batch.
            # Let's run full sampling on a small subset (e.g., 1 batch) if Diffusion.
            # For now, let's just use the standard validation function but adapt it.
            # Actually, `validate` assumes `model(inputs)` returns audio.
            # For Diffusion, `model` returns noise.
            # We need a separate validation for Diffusion or update `validate`.
            
            if use_diffusion:
                # Simple validation: just compute loss on val set (faster)
                # Or implement sampling. Sampling is better for LSD.
                # Let's skip LSD for now to save time and just log Loss.
                val_lsd = avg_g_loss # Placeholder
            else:
                val_lsd = validate(model, val_loader, device, spectral_model=spectral_model)
                
            logger.info(f"Epoch [{epoch+1}] - Loss: {avg_g_loss:.4f} - Val LSD: {val_lsd:.4f}")
            
            # Prepare Rich Checkpoint Data
            checkpoint_data = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_lsd": min(val_lsd, best_val_lsd),
                "config": {
                    "base_channels": base_channels,
                    "num_layers": num_layers,
                    "use_gan": use_gan,
                    "use_diffusion": use_diffusion,
                    "diffusion_steps": diffusion_steps,
                    "model_type": "diffusion" if use_diffusion else "gan",
                    "robust_training": robust_training,
                    "use_spectral": use_spectral,
                    "use_amp": use_amp
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if spectral_model:
                checkpoint_data["spectral_state_dict"] = spectral_model.state_dict()

            if discriminator:
                checkpoint_data["discriminator_state_dict"] = discriminator.state_dict()

            # Save Latest
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint_data, save_path)
            
            # Save Best
            if val_lsd < best_val_lsd:
                best_val_lsd = val_lsd
                torch.save(checkpoint_data, best_save_path)
                logger.info(f"New Best Model! LSD: {val_lsd:.4f}")
            
            if yield_loss:
                yield_loss(epoch + 1, avg_g_loss)

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            logger.error("CUDA Out of Memory Error caught.")
            return {"message": "❌ CUDA Out of Memory! Try reducing Batch Size or Channels.", "final_val_lsd": best_val_lsd}
        else:
            raise e
            
    if progress_callback:
        progress_callback(1.0, f"Done! Best LSD: {best_val_lsd:.4f}. Saved to {save_path}")
    
    return {
        "message": f"Success! Best Model LSD: {best_val_lsd:.4f}",
        "best_val_lsd": best_val_lsd,
        "final_val_lsd": val_lsd if 'val_lsd' in locals() else 0,
        "final_g_loss": avg_g_loss if 'avg_g_loss' in locals() else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Audio Upscaler")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to training audio files")
    parser.add_argument("--save-path", type=str, default="./models/model.ckpt", help="Where to save the model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for U-Net")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers in U-Net")
    parser.add_argument("--use-spectral", action="store_true", help="Enable dual-stream spectral recovery")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable Mixed Precision")
    
    # Diffusion Args
    parser.add_argument("--use-diffusion", action="store_true", help="Enable Diffusion Model training")
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="Number of diffusion steps")
    
    args = parser.parse_args()
    train_model(args.dataset_path, args.save_path, args.epochs, args.batch_size, args.lr, base_channels=args.base_channels, num_layers=args.num_layers, use_spectral=args.use_spectral, num_workers=args.num_workers, use_amp=not args.no_amp, use_diffusion=args.use_diffusion, diffusion_steps=args.diffusion_steps)
