import torch
import psutil
import logging

logger = logging.getLogger(__name__)

def get_system_info():
    """
    Detects system hardware capabilities (GPU VRAM, CPU Cores).
    Returns a dictionary with system stats.
    """
    info: dict = {
        "device_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

    if torch.cuda.is_available():
        try:
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            info["device_name"] = props.name
            info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
            
            # Get free memory (approximate)
            # Note: torch.cuda.mem_get_info() returns (free, total) in bytes
            free_bytes, _ = torch.cuda.mem_get_info(device_id)
            info["vram_free_gb"] = round(free_bytes / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to get CUDA details: {e}")
            
    return info

def get_training_recommendations(info=None):
    """
    Returns recommended training parameters based on hardware.
    """
    if info is None:
        info = get_system_info()
        
    vram = info["vram_total_gb"]
    
    # Defaults for CPU
    recs = {
        "batch_size": 2,
        "base_channels": 16,
        "num_layers": 3,
        "device": "CPU",
        "reason": "CPU detected. Using minimal settings to ensure runnable state."
    }
    
    if torch.cuda.is_available():
        recs["device"] = "CUDA"
        
        if vram < 4.0:
            recs.update({
                "batch_size": 2,
                "base_channels": 24,
                "num_layers": 3,
                "reason": f"Low VRAM ({vram}GB). Optimized for stability."
            })
        elif vram < 8.0:
            recs.update({
                "batch_size": 4,
                "base_channels": 32,
                "num_layers": 4,
                "reason": f"Medium VRAM ({vram}GB). Standard balanced settings."
            })
        elif vram < 16.0:
            recs.update({
                "batch_size": 8,
                "base_channels": 48,
                "num_layers": 5,
                "reason": f"High VRAM ({vram}GB). High-quality settings enabled."
            })
        else: # > 16GB
            recs.update({
                "batch_size": 16,
                "base_channels": 64,
                "num_layers": 6,
                "reason": f"Pro VRAM ({vram}GB). Maximum quality settings."
            })
            
    return recs

def estimate_vram_usage(batch_size, base_channels, num_layers, segment_length=16384, use_spectral=False, use_diffusion=False):
    """
    Estimates VRAM usage in GB for the AudioSuperResNet (+ SpectralUNet + Diffusion).
    Based on parameter count + activation sizes + optimizer states.
    """
    # Constants
    BYTES_PER_PARAM = 4 # Float32
    BYTES_PER_ACT = 4   # Float32
    OPTIMIZER_OVERHEAD = 2 # Adam keeps 2 states per param
    GRADIENT_OVERHEAD = 1 # Gradients
    
    # 1. Parameter Count Estimation
    # This is a rough heuristic based on the UNet architecture
    # Encoder: num_layers blocks. Channels double each time.
    # Block = 4 Conv1d(k=3) + 2 Conv1d(k=1) ~ 6 Convs
    # Param per Conv ~ C_in * C_out * K
    
    total_params = 0
    curr_c = base_channels
    
    # Head
    total_params += 1 * base_channels * 7
    
    # Encoder
    for _ in range(num_layers):
        # 2 Blocks * (4 Convs * 3x3 + 2 Convs * 1x1)
        # Approx 2 * (4 * c*c*3 + 2 * c*c*1) = 2 * (14 * c^2) = 28 * c^2
        total_params += 28 * (curr_c ** 2)
        # Downsample
        total_params += curr_c * (curr_c * 2) * 4
        curr_c *= 2
        
    # Bottleneck
    # 3 Blocks
    total_params += 3 * (28 * (curr_c ** 2))
    
    # Decoder
    for _ in range(num_layers):
        # Upsample
        total_params += curr_c * (curr_c // 2) * 4
        curr_c //= 2
        # 2 Blocks
        total_params += 28 * (curr_c ** 2)
        
    # Tail
    total_params += curr_c * 1 * 7
    
    # 2. Activation Memory Estimation
    # Sum of output sizes of all layers * batch_size
    # Length halves in encoder, doubles in decoder
    
    total_acts = 0
    curr_c = base_channels
    curr_l = segment_length
    
    # Head
    total_acts += curr_c * curr_l
    
    # Encoder
    for _ in range(num_layers):
        # 2 Blocks (internal activations)
        # Each block has ~6 layers of depth
        total_acts += 2 * 6 * (curr_c * curr_l)
        # Downsample
        curr_c *= 2
        curr_l //= 2
        total_acts += curr_c * curr_l
        
    # Bottleneck
    total_acts += 3 * 6 * (curr_c * curr_l)
    
    # Decoder
    for _ in range(num_layers):
        # Upsample
        curr_c //= 2
        curr_l *= 2
        total_acts += curr_c * curr_l
        # 2 Blocks
        total_acts += 2 * 6 * (curr_c * curr_l)
        
    # Total Memory = (Params * (1 + Opt + Grad)) + (Acts * Batch)
    # Plus some CUDA context overhead (~500MB)
    
    param_mem = total_params * BYTES_PER_PARAM * (1 + OPTIMIZER_OVERHEAD + GRADIENT_OVERHEAD)
    act_mem = total_acts * batch_size * BYTES_PER_ACT
    
    # Spectral Model Overhead (Heuristic)
    # 2D UNet is roughly 1.5x heavier in params and acts for similar dimensions
    if use_spectral:
        # We assume SpectralUNet is roughly similar in complexity to the Waveform UNet
        # but operates on 2D data which can be denser.
        # Adding 1.5x overhead is a safe conservative estimate.
        param_mem *= 2.5 # Base + 1.5x for Spectral
        act_mem *= 2.5
    
    # Diffusion Model Overhead
    # DiffusionUNet wrapper adds:
    # - Time embedding MLP (2-layer: time_dim(32) -> 128 -> 128)
    # - Additional concatenation/projection layers
    # - Scheduler (noise schedules, ~50MB)
    if use_diffusion:
        # Time embedding: ~20K params (32*128 + 128*128 = 20K)
        # Plus wrapper overhead for concatenation/processing
        diffusion_param_overhead = 0.15  # 15% increase
        param_mem *= (1 + diffusion_param_overhead)
        
        # Additional activation memory for timestep embeddings and intermediate noisy states
        act_mem *= 1.1
    
    # Safety buffer (CUDA context, fragmentation)
    buffer_mem = 0.5 * (1024**3) 
    
    total_bytes = param_mem + act_mem + buffer_mem
    return round(total_bytes / (1024**3), 2)

def check_vram_requirements(batch_size, base_channels, num_layers, segment_length=16384, use_spectral=False, use_diffusion=False):
    """
    Checks if the configuration fits in available VRAM.
    Returns (is_safe, message, estimated_gb, available_gb).
    """
    info = get_system_info()
    if info["device_name"] == "CPU":
        return True, "Running on CPU. VRAM check skipped.", 0, 0
        
    est = estimate_vram_usage(batch_size, base_channels, num_layers, segment_length, use_spectral, use_diffusion)
    avail = info["vram_free_gb"]
    
    # Allow if estimate is within 90% of available
    if est < (avail * 0.9):
        return True, f"Safe. Est: {est}GB / Free: {avail}GB", est, avail
    else:
        return False, f"⚠️ Risk of OOM! Est: {est}GB / Free: {avail}GB", est, avail
