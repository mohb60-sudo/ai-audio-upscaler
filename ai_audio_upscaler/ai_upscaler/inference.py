import torch
import logging
import os
from ..config import UpscalerConfig
from .model import AudioSuperResNet
from .diffusion import DiffusionScheduler, DiffusionUNet
from .spectral_model import SpectralUNet
from .discriminator import MultiResolutionDiscriminator
from ..post_processing import AudioPostProcessor

logger = logging.getLogger(__name__)

class AIUpscalerWrapper:
    """
    Wrapper class for the AudioSuperResNet model to handle inference.
    
    Responsibilities:
    - Model initialization and device management (CPU/CUDA)
    - Checkpoint loading
    - Chunked inference to prevent Out-Of-Memory (OOM) errors on large files
    - Data preparation (dimension handling)
    """
    def __init__(self, config: UpscalerConfig):
        self.config = config
        
        # Determine device
        if config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if config.device == 'cuda':
                logger.warning("CUDA requested but not available. Falling back to CPU.")
        
        logger.info(f"AI Model initialized on device: {self.device}")
        
        # Initialize Spectral Model (Optional)
        self.spectral_model = None
        self.discriminator = None
        self.use_diffusion = False
        self.diffusion_scheduler = None
        self.separator = None
        self.restorer = None # Added for stem restoration
        
        # Load Checkpoint & Init Model
        if config.model_checkpoint and os.path.exists(config.model_checkpoint):
            logger.info(f"Loading model checkpoint: {config.model_checkpoint}")
            try:
                logger.info(f"✅ ATTEMPTING TO LOAD CHECKPOINT: {config.model_checkpoint}")
                checkpoint = torch.load(config.model_checkpoint, map_location=self.device)
                logger.info(f"✅ SUCCESSFULLY LOADED CHECKPOINT: {config.model_checkpoint}")
                
                # Default Params
                base_channels = 32
                num_layers = 4
                use_spectral = False
                use_diffusion = False
                diffusion_steps = 1000
                
                # Handle Rich Checkpoint
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    model_conf = checkpoint["config"]
                    base_channels = model_conf.get("base_channels", 32)
                    num_layers = model_conf.get("num_layers", 4)
                    use_spectral = model_conf.get("use_spectral", False)
                    use_diffusion = model_conf.get("use_diffusion", False)
                    diffusion_steps = model_conf.get("diffusion_steps", 1000)
                    logger.info(f"Found Rich Checkpoint. Config: {base_channels}ch / {num_layers}L / Spectral={use_spectral} / Diffusion={use_diffusion}")
                    
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # Intermediate format (rare)
                    state_dict = checkpoint["state_dict"]
                else:
                    # Legacy Format
                    state_dict = checkpoint
                    logger.warning("Legacy checkpoint detected. Assuming default architecture (32ch/4L).")

                self.use_diffusion = use_diffusion

                # Initialize Model with correct config
                if use_diffusion:
                    # Diffusion Model: Base model takes 2 channels (Noisy Input + Condition)
                    base_model = AudioSuperResNet(in_channels=2, base_channels=base_channels, num_layers=num_layers, time_emb_dim=32)
                    self.model = DiffusionUNet(base_model)
                    self.diffusion_scheduler = DiffusionScheduler(num_steps=diffusion_steps)
                else:
                    self.model = AudioSuperResNet(in_channels=1, base_channels=base_channels, num_layers=num_layers)
                
                self.model.to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                # Initialize Spectral Model if enabled in checkpoint
                if use_spectral and "spectral_state_dict" in checkpoint:
                    logger.info("Initializing Spectral Recovery Module...")
                    self.spectral_model = SpectralUNet(in_channels=1, base_channels=base_channels)
                    self.spectral_model.to(self.device)
                    self.spectral_model.load_state_dict(checkpoint["spectral_state_dict"])
                    self.spectral_model.eval()

                # Initialize Discriminator if available (for QC)
                if "discriminator_state_dict" in checkpoint:
                    logger.info("Initializing Discriminator for Quality Control...")
                    self.discriminator = MultiResolutionDiscriminator()
                    self.discriminator.to(self.device)
                    self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
                    self.discriminator.eval()
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                # Fallback to default if load fails (to prevent crash, but warn heavily)
                self.model = AudioSuperResNet(in_channels=1, base_channels=32, num_layers=4)
                self.model.to(self.device)
                self.model.eval()
        else:
            logger.warning("No valid checkpoint provided. Using RANDOM INITIALIZED weights (32ch/4L). Output will be noisy!")
            self.model = AudioSuperResNet(in_channels=1, base_channels=32, num_layers=4)
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def list_available_models(models_dir="models"):
        """Scans the models directory for .pth or .ckpt files."""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        return [f for f in os.listdir(models_dir) if f.endswith(".pth") or f.endswith(".ckpt")]

    def estimate_max_batch_size(self, chunk_samples: int, channels: int = 1, vram_limit_ratio: float = 0.8) -> int:
        """
        Estimates the maximum safe batch size based on available VRAM and Model Complexity.
        """
        if self.device.type != 'cuda':
            return 1 # CPU is slow, keep batch size 1
            
        try:
            # Get free memory in bytes
            free_mem, total_mem = torch.cuda.mem_get_info(self.device)
            
            # Enforce Limit (User requested max 80% load)
            max_allowed = total_mem * vram_limit_ratio
            allocated = total_mem - free_mem # Roughly
            effective_free = max(0, max_allowed - allocated)
            
            # Use the tighter constraint (Real Free vs Limit Free)
            available_mem = min(free_mem, effective_free)
            
            # Reserve 1.5GB for safety (OS, Display, Gradio overhead)
            safety_margin = 1.5 * 1024**3
            available_mem = max(0, available_mem - safety_margin)
            
            # Calculate Model Complexity Factor
            # Base cost for 32ch/4L ~ 4KB/sample
            # Scaling: (ch/32)^2 * (layers/4)
            
            # Get model config (handle DiffusionUNet wrapper)
            if hasattr(self.model, "base_model"):
                # DiffusionUNet -> AudioSuperResNet
                cfg_ch = self.model.base_model.base_channels
                cfg_layers = self.model.base_model.num_layers
            elif hasattr(self.model, "base_channels"):
                # AudioSuperResNet
                cfg_ch = self.model.base_channels
                cfg_layers = self.model.num_layers
            else:
                cfg_ch = 32
                cfg_layers = 4
                
            complexity_factor = ((cfg_ch / 32) ** 2) * (cfg_layers / 4)
            
            # Diffusion models need more memory for intermediate states
            if self.use_diffusion:
                complexity_factor *= 1.5
            
            bytes_per_sample = 4096 * complexity_factor
            
            cost_per_item = chunk_samples * bytes_per_sample * channels
            
            if cost_per_item <= 0: return 1
            
            max_batch = int(available_mem / cost_per_item)
            
            logger.info(f"VRAM Est: Free={free_mem/1024**3:.2f}GB, Limit={max_allowed/1024**3:.2f}GB, Avail={available_mem/1024**3:.2f}GB, MaxBatch={max_batch}")
            
            return max(1, min(max_batch, 32))
            
        except Exception as e:
            logger.warning(f"Failed to estimate VRAM: {e}. Defaulting to batch size 1.")
            return 1

    def enhance(self, waveform: torch.Tensor, original_sr: int, target_sr: int, chunk_seconds: float = 2.0,
                tta: bool = False, stereo_mode: str = "lr", transient_strength: float = 0.0, spectral_matching: bool = False,
                qc: bool = False, candidate_count: int = 8, judge_threshold: float = 0.5, diffusion_steps: int = 50,

                denoising_strength: float = 0.6, noise_scale: float = 1.1, progress_callback=None, 
                use_stems: bool = False, use_restoration: bool = False, restoration_strength: float = 0.5) -> torch.Tensor:
        """
        Main entry point for audio enhancement with Overlap-Add (OLA) and VRAM management.
        """
        # Stem-Aware Upscaling
        if use_stems:
            logger.info("Stem-Aware Upscaling Enabled. Initializing Separator...")
            if self.separator is None:
                from ..separation import SourceSeparator
                self.separator = SourceSeparator(self.device)
            
            # Separate
            stems = self.separator.separate(waveform, original_sr)
            
            # Intelligent Restoration
            if use_restoration:
                logger.info("Applying Intelligent Stem Restoration...")
                if self.restorer is None:
                    from ..restoration import StemRestorer
                    self.restorer = StemRestorer(self.device)
                
                for name, stem in stems.items():
                    # Analyze
                    profile = self.restorer.analyze_stem(stem, original_sr, name)
                    # Restore
                    stems[name] = self.restorer.restore_stem(stem, original_sr, name, profile, restoration_strength)
            
            # Unload separator to free VRAM for upscaling
            self.separator.unload_model()
            
            enhanced_stems = []
            for name, stem in stems.items():
                logger.info(f"Upscaling Stem: {name}...")
                # Recursive call with use_stems=False
                enhanced = self.enhance(
                    stem, original_sr, target_sr, chunk_seconds,
                    tta, stereo_mode, transient_strength, spectral_matching,
                    qc, candidate_count, judge_threshold, diffusion_steps,
                    denoising_strength, noise_scale, progress_callback,

                    use_stems=False, # CRITICAL: Prevent infinite recursion
                    use_restoration=False # Don't restore already restored stems
                )
                enhanced_stems.append(enhanced)
            
            # Sum stems
            logger.info("Recombining Stems...")
            out = torch.stack(enhanced_stems).sum(dim=0)
            return out

        # Redirect to QC logic if enabled
        if qc:
            return self.enhance_with_qc(
                waveform, original_sr, target_sr, chunk_seconds,
                tta, stereo_mode, transient_strength, spectral_matching,
                candidate_count, judge_threshold
            )

        # Clamp input to [-1, 1]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Global Normalization
        global_max = torch.max(torch.abs(waveform))
        if global_max > 1e-8:
            logger.info(f"Global Normalization: Peak {global_max:.4f} -> 1.0")
            waveform = waveform / global_max
        else:
            global_max = torch.tensor(1.0, device=self.device)

        channels, time = waveform.shape
        chunk_size = int(target_sr * chunk_seconds)
        
        # Determine batch size
        max_items = self.estimate_max_batch_size(chunk_size, channels)
        
        if tta or stereo_mode == "ms":
            batch_size = 1
        else:
            batch_size = max(1, max_items // channels) # Split channels across batch if needed? No, usually (B, C, T)
            # Actually, if we process (B, C, T), cost is per item.
            batch_size = max(1, max_items)

        # Overlap-Add Configuration
        overlap_percent = 0.25
        overlap_size = int(chunk_size * overlap_percent)
        stride = chunk_size - overlap_size
        
        # If short enough, run directly (no OLA needed)
        if time <= chunk_size:
            out = self._enhance_batch(
                waveform.unsqueeze(0), 
                tta=tta, 
                stereo_mode=stereo_mode, 
                diffusion_steps=diffusion_steps,
                denoising_strength=denoising_strength,
                noise_scale=noise_scale,
                progress_callback=progress_callback,
                current_chunk_idx=0,
                total_chunks=1
            )
            out = out.squeeze(0)
        else:
            logger.info(f"Audio length {time} > chunk {chunk_size}. Using OLA (Overlap: {overlap_size}, Batch: {batch_size}).")
            
            # Create Output Buffer
            output_tensor = torch.zeros((channels, time), device=self.device)
            weight_tensor = torch.zeros((channels, time), device=self.device)
            
            # Create Window Function (Hanning)
            window = torch.hann_window(chunk_size, device=self.device).view(1, -1) # (1, T)
            
            # Prepare Chunks
            chunks = []
            coords = [] # (start, end)
            
            # 1. Slice Chunks
            for start in range(0, time, stride):
                end = min(start + chunk_size, time)
                chunk_len = end - start
                
                chunk = waveform[:, start:end]
                
                # Pad if needed (for last chunk)
                if chunk_len < chunk_size:
                    padding = chunk_size - chunk_len
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                chunks.append(chunk)
                coords.append((start, end))
            
            total_chunks = len(chunks)
            
            # 2. Process Batches
            for i in range(0, total_chunks, batch_size):
                # Garbage Collection before batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                batch_indices = range(i, min(i + batch_size, total_chunks))
                batch_list = [chunks[j] for j in batch_indices]
                
                batch_tensor = torch.stack(batch_list)
                
                # Inference
                processed_batch = self._enhance_batch(
                    batch_tensor, 
                    tta=tta, 
                    stereo_mode=stereo_mode, 
                    diffusion_steps=diffusion_steps,
                    denoising_strength=denoising_strength,
                    noise_scale=noise_scale,
                    progress_callback=progress_callback,
                    current_chunk_idx=i,
                    total_chunks=total_chunks
                )
                
                # 3. Accumulate (OLA)
                for k, processed in enumerate(processed_batch):
                    idx = i + k
                    start, end = coords[idx]
                    valid_len = end - start
                    
                    # Apply Window
                    processed_windowed = processed * window
                    
                    # Trim padding
                    if valid_len < chunk_size:
                        processed_windowed = processed_windowed[:, :valid_len]
                        this_window = window[:, :valid_len]
                    else:
                        this_window = window
                        
                    # Add to buffer
                    output_tensor[:, start:end] += processed_windowed
                    weight_tensor[:, start:end] += this_window
            
            # 4. Normalize by Weights
            # Avoid division by zero
            weight_tensor = torch.clamp(weight_tensor, min=1e-8)
            out = output_tensor / weight_tensor
            
        # Global Denormalization
        if global_max > 1e-8:
            out = out * global_max.to(out.device)

        if transient_strength > 0:
            logger.info(f"Applying Transient Restoration (Strength: {transient_strength})")
            out = AudioPostProcessor.restore_transients(waveform, out, strength=transient_strength)
            
        # Spectral Matching
        if spectral_matching:
            logger.info("Applying Spectral Matching...")
            out = AudioPostProcessor.match_spectral_balance(out)
            
        return out

    def enhance_with_qc(self, waveform: torch.Tensor, original_sr: int, target_sr: int, chunk_seconds: float = 2.0,
                        tta: bool = False, stereo_mode: str = "lr", transient_strength: float = 0.0, spectral_matching: bool = False,
                        candidate_count: int = 8, judge_threshold: float = 0.5) -> torch.Tensor:
        """
        Runs inference with Hybrid Quality Control (Judge + Consensus).
        """
        logger.info(f"Starting Hybrid QC with {candidate_count} candidates...")
        
        channels, time = waveform.shape
        chunk_size = int(target_sr * chunk_seconds)
        
        # Determine optimal batch size
        max_batch = self.estimate_max_batch_size(chunk_size)
        
        # MS Processing requires exactly 2 channels, so we can't easily batch multiple candidates
        if stereo_mode == "ms":
            max_batch = 1
            
        logger.info(f"QC Mode: Generating {candidate_count} candidates. Max Batch Size: {max_batch}")
        
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp(prefix="qc_candidates_")
        candidate_paths = []
        
        try:
            # Generate candidates sequentially
            for i in range(candidate_count):
                logger.info(f"QC: Generating candidate {i+1}/{candidate_count}...")
                
                # Run inference
                candidate = self.enhance(
                    waveform, original_sr, target_sr, chunk_seconds,
                    tta=tta, stereo_mode=stereo_mode, 
                    transient_strength=0.0, # Disable for candidates
                    spectral_matching=False, # Disable for candidates
                    qc=False 
                )
                
                # Save to disk immediately
                path = os.path.join(temp_dir, f"candidate_{i}.pt")
                # Clamp to ensure valid audio range for discriminator
                candidate = torch.clamp(candidate, -1.0, 1.0)
                torch.save(candidate.cpu(), path)
                candidate_paths.append(path)
                
                # Free memory
                del candidate
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            # --- Filter A: Judge (Discriminator) ---
            if self.discriminator:
                logger.info("QC: Applying Judge Filter (Streaming)...")
                scored_paths = []
                
                for path in candidate_paths:
                    # Load -> GPU -> Score -> Unload
                    cand = torch.load(path, map_location=self.device)
                    
                    # Use chunked scoring to prevent OOM
                    score = self.discriminator.score_candidate(cand, chunk_seconds=chunk_seconds, sr=target_sr)
                    
                    scored_paths.append((path, score))
                    del cand
                    
                    # Explicit cleanup
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Filter
                passed_paths = [p for p, s in scored_paths if s >= judge_threshold]
                
                logger.info(f"QC Judge: {len(passed_paths)}/{len(candidate_paths)} candidates passed threshold {judge_threshold}")
                
                if not passed_paths:
                    logger.warning("QC Judge rejected ALL candidates! Fallback to all.")
                    passed_paths = candidate_paths
                else:
                    avg_score = sum(s for p, s in scored_paths if s >= judge_threshold) / len(passed_paths)
                    logger.info(f"QC Judge: Average Score of Passed: {avg_score:.4f}")
                    
                candidate_paths = passed_paths
            else:
                logger.info("QC: Discriminator not available. Skipping Judge Filter.")

            # --- Filter B: Consensus (Variance Rejection) ---
            # Streaming implementation to avoid loading all files
            if len(candidate_paths) > 2:
                logger.info("QC: Applying Consensus Filter (Streaming)...")
                
                # 1. Compute Mean and Variance (Welford's Algorithm)
                # Initialize
                n = 0
                mean_accum = None
                m2_accum = None # Sum of squares of differences from the current mean
                
                for path in candidate_paths:
                    x = torch.load(path, map_location='cpu')
                    n += 1
                    
                    if mean_accum is None:
                        mean_accum = torch.zeros_like(x)
                        m2_accum = torch.zeros_like(x)
                        mean_accum[:] = x
                    else:
                        delta = x - mean_accum
                        mean_accum += delta / n
                        delta2 = x - mean_accum
                        m2_accum += delta * delta2
                    
                    del x
                
                # Finalize Variance
                if n < 2:
                    variance = torch.zeros_like(mean_accum)
                else:
                    variance = m2_accum / (n - 1)
                    
                mse_std = variance.mean().sqrt()
                mse_mean = variance.mean() # MSE of candidates from the mean is roughly the variance
                
                # Wait, MSE logic in original was: mean((cand - mean_waveform)**2)
                # Which IS the variance at each pixel, averaged over pixels?
                # Yes. variance.mean() gives the average pixel variance.
                
                if mse_std < 1e-8:
                    logger.info("QC Consensus: Variance is negligible. Skipping filter.")
                    out = mean_accum
                else:
                    rejection_threshold = mse_mean + mse_std
                    
                    final_paths = []
                    
                    # 2. Filter Candidates
                    for path in candidate_paths:
                        x = torch.load(path, map_location='cpu')
                        mse = torch.mean((x - mean_accum) ** 2)
                        
                        if mse <= rejection_threshold:
                            final_paths.append(path)
                        else:
                            logger.info(f"QC Consensus: Rejected candidate (MSE: {mse:.2e} > {rejection_threshold:.2e})")
                        del x
                    
                    if not final_paths:
                        logger.warning("QC Consensus rejected all! Fallback to mean.")
                        out = mean_accum
                    else:
                        # 3. Compute Final Mean of Survivors
                        # Streaming again
                        n_final = 0
                        final_mean = torch.zeros_like(mean_accum)
                        
                        for path in final_paths:
                            x = torch.load(path, map_location='cpu')
                            final_mean += x
                            n_final += 1
                            del x
                            
                        out = final_mean / n_final
            else:
                # Just average
                n = 0
                out = None
                for path in candidate_paths:
                    x = torch.load(path, map_location='cpu')
                    if out is None:
                        out = torch.zeros_like(x)
                    out += x
                    n += 1
                    del x
                out = out / n
                
            # Move result back to GPU
            out = out.to(self.device)
            
        finally:
            # Cleanup Temp Files
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"QC: Cleaned up temp directory {temp_dir}")
            
        # --- Global Post-Processing ---
        # Apply Transient Restoration and Spectral Matching to the FINAL merged result
        if transient_strength > 0:
            logger.info(f"Applying Transient Restoration (Strength: {transient_strength})")
            out = AudioPostProcessor.restore_transients(waveform, out, strength=transient_strength)
            
        if spectral_matching:
            logger.info("Applying Spectral Matching...")
            out = AudioPostProcessor.match_spectral_balance(out)
            
        return out

    def _enhance_batch(self, batch_waveform: torch.Tensor, tta: bool = False, stereo_mode: str = "lr", 
                       diffusion_steps: int = 50, denoising_strength: float = 0.6, noise_scale: float = 1.1, progress_callback=None, current_chunk_idx=0, total_chunks=1) -> torch.Tensor:
        """
        Helper to run inference on a batch of chunks.
        Args:
            batch_waveform: (Batch, Channels, Time)
            denoising_strength: 0.0 to 1.0. Controls starting timestep.
        """
        with torch.no_grad():
            batch_device = batch_waveform.to(self.device)
            batch_size, channels, time = batch_device.shape
            
            # Define the core model function (what runs on the GPU)
            def model_fn(x_input):
                # x_input is (Batch, Channels, Time)
                # Reshape for model: (Batch*Channels, 1, Time)
                b, c, time_dim = x_input.shape
                x_flat = x_input.view(b * c, 1, time_dim)
                
                try:
                    if self.use_diffusion and self.diffusion_scheduler:
                        # Diffusion Inference
                        
                        # Calculate starting timestep based on strength
                        # strength 1.0 -> start at t=num_steps-1 (Full Noise)
                        # strength 0.0 -> start at t=0 (No Noise)
                        # strength 0.5 -> start at t=num_steps//2
                        
                        # Calculate absolute max step
                        max_step = self.diffusion_scheduler.num_steps - 1
                        start_t_absolute = int(max_step * denoising_strength)
                        start_t_absolute = max(0, min(start_t_absolute, max_step))
                        
                        # Ensure diffusion_steps does not exceed model capacity
                        effective_steps = diffusion_steps
                        if effective_steps > self.diffusion_scheduler.num_steps:
                            logger.warning(f"Requested steps {effective_steps} > Model steps {self.diffusion_scheduler.num_steps}. Clamping.")
                            effective_steps = self.diffusion_scheduler.num_steps
                        
                        # Generate timesteps for strided sampling
                        step_ratio = max(1, self.diffusion_scheduler.num_steps // effective_steps)
                        
                        timesteps = []
                        for i in range(effective_steps):
                            t = i * step_ratio
                            # Only include steps <= start_t_absolute
                            if t <= start_t_absolute:
                                timesteps.append(t)
                        
                        if not timesteps:
                            # Strength too low, just return input
                            return x_flat
                            
                        # Start from the highest timestep in our filtered list
                        start_t = timesteps[-1]
                        
                        # Initialize x with Partial Diffusion (Noisy Input)
                        # Instead of pure noise, we start with the input diffused to start_t
                        t_start_batch = torch.full((x_flat.shape[0],), start_t, device=self.device, dtype=torch.long)
                        noise = torch.randn_like(x_flat)
                        x, _ = self.diffusion_scheduler.add_noise(x_flat, t_start_batch, noise)
                        
                        total_steps = len(timesteps)
                        
                        for step_idx, t_idx in enumerate(reversed(timesteps)):
                            # Calculate t_prev (next step in reverse)
                            # We need to find the previous step in our filtered list
                            # timesteps is [0, 2, ..., start_t]
                            # reversed is [start_t, ..., 2, 0]
                            
                            # Current index in the original list?
                            # No, just look at the list.
                            
                            current_idx_in_list = len(timesteps) - 1 - step_idx
                            if current_idx_in_list > 0:
                                t_prev_val = timesteps[current_idx_in_list - 1]
                            else:
                                t_prev_val = -1
                                
                            # Progress Reporting (every 5 steps)
                            if progress_callback and step_idx % 5 == 0:
                                batch_progress = step_idx / total_steps
                                global_p = (current_chunk_idx + (batch_progress * batch_size)) / total_chunks
                                ui_progress = 0.5 + (global_p * 0.3)
                                
                                # Format: Step X/Y (Chunk A-B/Z)
                                chunk_range = f"{current_chunk_idx+1}"
                                if batch_size > 1:
                                    chunk_range += f"-{min(current_chunk_idx+batch_size, total_chunks)}"
                                
                                progress_callback(ui_progress, f"Diffusion: Step {step_idx+1}/{total_steps} (Chunk {chunk_range}/{total_chunks})")

                            t_batch = torch.full((x.shape[0],), t_idx, device=self.device, dtype=torch.long)
                            t_prev_batch = torch.full((x.shape[0],), t_prev_val, device=self.device, dtype=torch.long)
                            
                            pred_noise = self.model(x, t_batch, condition=x_flat)
                            
                            # Use DDIM if skipping steps, otherwise standard DDPM
                            if step_ratio > 1:
                                x = self.diffusion_scheduler.step_ddim(pred_noise, t_batch, x, t_prev_batch, noise_scale=noise_scale)
                            else:
                                x = self.diffusion_scheduler.step(pred_noise, t_batch, x)
                            
                        out = x
                    else:
                        out = self.model(x_flat)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error("OOM Error inside chunk. Try reducing chunk_seconds.")
                        raise e
                    raise e
                
                # Reshape back to (Batch, Channels, Time)
                return out.view(b, c, time_dim)

            # --- Apply Post-Processing Wrappers ---
            # Note: TTA/MS logic below assumes model_fn handles the batch dimension correctly.
            # If TTA/MS is enabled, we forced batch_size=1 in enhance(), so x_input is (1, C, T).
            # This ensures compatibility with existing AudioPostProcessor logic.
            
            # 1. Mid-Side Processing
            if stereo_mode == "ms" and channels == 2:
                if tta:
                    def tta_wrapper(x):
                        return AudioPostProcessor.apply_tta(model_fn, x, self.device)
                    out = AudioPostProcessor.process_mid_side(tta_wrapper, batch_device, self.device)
                else:
                    out = AudioPostProcessor.process_mid_side(model_fn, batch_device, self.device)
            
            # 2. Standard Processing (L/R or Mono)
            else:
                if tta:
                    out = AudioPostProcessor.apply_tta(model_fn, batch_device, self.device)
                else:
                    out = model_fn(batch_device)
            
            # Clamp to valid audio range [-1, 1]
            out = torch.clamp(out, -1.0, 1.0)
            
            return out
