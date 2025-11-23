import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import glob
import logging
from ai_audio_upscaler.ai_upscaler.model import AudioSuperResNet
from ai_audio_upscaler.ai_upscaler.transforms import MP3Compression, BandwidthLimiter, QuantizationNoise
from ai_audio_upscaler.dsp import DSPUpscaler

# Configure logging
logger = logging.getLogger(__name__)

def find_audio_files(data_dir):
    """Recursively finds all supported audio files in a directory."""
    extensions = ['*.wav', '*.flac', '*.mp3']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(files)

class RobustAudioDataset(Dataset):
    """
    Robust dataset loader for Audio Super Resolution.
    - Supports WAV, FLAC, MP3.
    - Auto-resamples to target_sr (Ground Truth).
    - Downsamples to input_sr (Input).
    """
    def __init__(self, data_dir, target_sr=48000, input_sr=24000, segment_length=16384):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_length = segment_length
        
        self.files = find_audio_files(data_dir)
            
        if not self.files:
            logger.warning(f"No audio files found in {data_dir}")
        else:
            logger.info(f"Found {len(self.files)} training files.")

        # Pre-instantiate resamplers (will be cloned/handled in getitem if needed, 
        # but torchaudio transforms are usually stateless or we recreate them)
        # Actually, for variable input SRs, we need to resample dynamically.
        
        # We need a baseline upscaler for the input features
        self.dsp = DSPUpscaler(target_sample_rate=target_sr, method="sinc")
        
        # Augmentations
        self.mp3_aug = MP3Compression(sample_rate=target_sr)
        self.bw_aug = BandwidthLimiter(sample_rate=target_sr)
        self.quant_aug = QuantizationNoise()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            waveform, sr = torchaudio.load(path)
            
            # 1. Convert to Mono (for prototype simplicity)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 2. Resample to Target SR (Ground Truth)
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
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
            
            # Apply Bandwidth Limiting (simulates low-pass filters in compression)
            degraded = self.bw_aug(target)
            
            # Apply Quantization Noise (simulates low bit depth/bitrate noise)
            degraded = self.quant_aug(degraded)
            
            # Apply MP3 Compression (simulates coding artifacts)
            degraded = self.mp3_aug(degraded)
            
            # Downsample to input SR
            downsampler = torchaudio.transforms.Resample(self.target_sr, self.input_sr)
            low_res = downsampler(degraded)
            
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

from ai_audio_upscaler.ai_upscaler.discriminator import MultiResolutionDiscriminator
from ai_audio_upscaler.ai_upscaler.loss import MultiResolutionSTFTLoss, feature_loss, generator_loss, discriminator_loss

from torch.utils.data import Dataset, DataLoader, random_split
from ai_audio_upscaler.ai_upscaler.metrics import calculate_lsd

def validate(model, val_loader, device):
    """
    Run validation on held-out set.
    Returns average LSD score (Lower is better).
    """
    model.eval()
    total_lsd = 0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
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

def train_model(data_dir, save_path, epochs=10, batch_size=16, lr=1e-4, device="cuda", use_gan=False, progress_callback=None, yield_loss=None):
    """
    Main training function callable from UI.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
        
    device = torch.device(device)
    logger.info(f"Training on {device} (GAN={use_gan})")
    
    if progress_callback: progress_callback(0, "Initializing Dataset...")
    
    full_dataset = RobustAudioDataset(data_dir)
    if len(full_dataset) == 0:
        if progress_callback: progress_callback(0, "Error: Dataset empty.")
        return "Dataset Empty"

    # Split Train/Val (90/10)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Dataset Split: {train_size} Train, {val_size} Val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Generator
    model = AudioSuperResNet().to(device)
    optimizer_g = optim.Adam(model.parameters(), lr=lr)
    
    # Discriminator (only if GAN enabled)
    discriminator = None
    optimizer_d = None
    if use_gan:
        discriminator = MultiResolutionDiscriminator().to(device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    stft_criterion = MultiResolutionSTFTLoss().to(device)

    # Load existing checkpoint if available (to continue training)
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            logger.info(f"Resumed from {save_path}")
        except:
            logger.warning("Could not load existing checkpoint. Starting fresh.")

    best_val_lsd = float('inf')
    best_save_path = save_path.replace(".ckpt", "_best.ckpt")

    for epoch in range(epochs):
        model.train()
        if discriminator: discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- Train Discriminator ---
            if use_gan and epoch >= int(epochs * 0.2): # Warmup
                optimizer_d.zero_grad()
                fake_audio = model(inputs).detach()
                real_scores, fake_scores, _, _ = discriminator(targets, fake_audio)
                d_loss, _, _ = discriminator_loss(real_scores, fake_scores)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
                total_d_loss += d_loss.item()
            
            # --- Train Generator ---
            optimizer_g.zero_grad()
            fake_audio = model(inputs)
            sc_loss = stft_criterion(fake_audio, targets)
            
            adv_loss = 0
            fm_loss = 0
            if use_gan and epoch >= int(epochs * 0.2):
                real_scores, fake_scores, fmap_r, fmap_g = discriminator(targets, fake_audio)
                adv_loss, _ = generator_loss(fake_scores)
                fm_loss = feature_loss(fmap_r, fmap_g)
                g_loss = 45 * sc_loss + 0.1 * adv_loss + 2 * fm_loss
            else:
                g_loss = sc_loss

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()
            total_g_loss += g_loss.item()
            
            if progress_callback and batch_idx % 5 == 0:
                overall_progress = (epoch + batch_idx / num_batches) / epochs
                msg = f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{num_batches} - G Loss: {g_loss.item():.4f}"
                progress_callback(overall_progress, msg)
        
        avg_g_loss = total_g_loss / num_batches
        
        # --- Validation Step ---
        val_lsd = validate(model, val_loader, device)
        logger.info(f"Epoch [{epoch+1}/{epochs}] - G Loss: {avg_g_loss:.4f} - Val LSD: {val_lsd:.4f}")
        
        # Save Latest
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        
        # Save Best
        if val_lsd < best_val_lsd:
            best_val_lsd = val_lsd
            torch.save(model.state_dict(), best_save_path)
            logger.info(f"New Best Model! LSD: {val_lsd:.4f}")
        
        if yield_loss:
            yield_loss(epoch + 1, avg_g_loss)

    if progress_callback: progress_callback(1.0, f"Done! Best LSD: {best_val_lsd:.4f}. Saved to {save_path}")
    return f"Success! Best Model LSD: {best_val_lsd:.4f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Audio Upscaler")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to training audio files")
    parser.add_argument("--save-path", type=str, default="./checkpoints/model.ckpt", help="Where to save the model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    train_model(args.dataset_path, args.save_path, args.epochs, args.batch_size, args.lr)
