import torch
import torch.nn.functional as F

def calculate_lsd(ref_spec, deg_spec):
    """
    Calculate Log-Spectral Distance (LSD) between two spectrograms.
    
    Args:
        ref_spec (torch.Tensor): Reference spectrogram (Magnitude).
        deg_spec (torch.Tensor): Degraded/Generated spectrogram (Magnitude).
        
    Returns:
        float: LSD value (Lower is better).
    """
    # Ensure inputs are tensors
    if not isinstance(ref_spec, torch.Tensor):
        ref_spec = torch.tensor(ref_spec)
    if not isinstance(deg_spec, torch.Tensor):
        deg_spec = torch.tensor(deg_spec)
        
    # Avoid log(0)
    ref_log = torch.log10(ref_spec + 1e-7)
    deg_log = torch.log10(deg_spec + 1e-7)
    
    diff = (ref_log - deg_log) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(diff, dim=-2))) # Mean over time, then freq
    
    return lsd.item()

def calculate_ssim(ref_spec, deg_spec):
    """
    Calculate Structural Similarity Index (SSIM) on spectrograms.
    Simplified implementation for 2D tensors.
    
    Args:
        ref_spec (torch.Tensor): Reference spectrogram.
        deg_spec (torch.Tensor): Degraded/Generated spectrogram.
        
    Returns:
        float: SSIM value (Higher is better, max 1.0).
    """
    # Normalize to 0-1 range for SSIM calculation
    min_val = min(ref_spec.min(), deg_spec.min())
    max_val = max(ref_spec.max(), deg_spec.max())
    
    ref_norm = (ref_spec - min_val) / (max_val - min_val + 1e-7)
    deg_norm = (deg_spec - min_val) / (max_val - min_val + 1e-7)
    
    # Add batch/channel dims for conv2d if needed
    if ref_norm.dim() == 2:
        ref_norm = ref_norm.unsqueeze(0).unsqueeze(0)
        deg_norm = deg_norm.unsqueeze(0).unsqueeze(0)
        
    # Basic SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(ref_norm, 3, 1, 1)
    mu_y = F.avg_pool2d(deg_norm, 3, 1, 1)
    
    sigma_x = F.avg_pool2d(ref_norm ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(deg_norm ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(ref_norm * deg_norm, 3, 1, 1) - mu_x * mu_y
    
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    ssim_map = ssim_n / ssim_d
    return torch.mean(ssim_map).item()
