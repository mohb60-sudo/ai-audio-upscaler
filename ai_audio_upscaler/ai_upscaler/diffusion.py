import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionScheduler:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Computes x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise.
        This allows us to jump directly to any timestep t without iterating.
        
        Args:
            x_start: Original clean audio x_0.
            t: Target timestep.
            noise: Optional noise tensor (if None, generated randomly).
            
        Returns:
            x_t: Noisy audio at timestep t.
            noise: The noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise
        
    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu().long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def step(self, model_output, t, x_t):
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        Using DDPM sampling.
        """
        # t is (Batch,) tensor of timesteps
        # We need to extract coefficients for each item in the batch
        
        # 1. Predict x_0 from model output (assuming model predicts noise 'epsilon')
        # x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        
        # Extract coefficients
        sqrt_one_minus_at = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_at = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        
        pred_x0 = (x_t - sqrt_one_minus_at * model_output) / sqrt_at
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) # Clip for stability
        
        # 2. Compute posterior mean
        # mean = coeff1 * pred_x0 + coeff2 * x_t
        
        # Extract posterior coefficients for the specific timesteps t
        post_mean_coef1 = self.extract(self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), t, x_t.shape)
        post_mean_coef2 = self.extract((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod), t, x_t.shape)
        
        posterior_mean = post_mean_coef1 * pred_x0 + post_mean_coef2 * x_t
        
        # 3. Add noise (if t > 0)
        # We need to handle the case where some items in batch might be at t=0 (though usually all are same t)
        # But for standard sampling, all batch items are at same t.
        # However, to be robust, we should use a mask.
        
        noise = torch.randn_like(x_t)
        
        # Extract posterior variance
        # We pre-calculated posterior_variance, but we need log variance for stability
        # posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        # Let's compute it on the fly or use the pre-calc property if we had it.
        # We'll compute it here to be safe and batched.
        
        # Note: self.posterior_variance is (Num_Steps,)
        # We need to extract for t
        post_var = self.extract(self.posterior_variance, t, x_t.shape)
        
        # Log variance for stability (clipping first element)
        # We can just use the extracted variance and log it, clamping min value
        post_log_var = torch.log(torch.clamp(post_var, min=1e-20))
        
        # Mask for t > 0
        # t is (Batch,)
        nonzero_mask = (t > 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        
        x_prev = posterior_mean + nonzero_mask * torch.exp(0.5 * post_log_var) * noise
            
        return x_prev

    @torch.no_grad()
    def step_ddim(self, model_output, t, x_t, t_prev, eta=0.0, noise_scale=1.0):
        """
        DDIM (Denoising Diffusion Implicit Models) Sampling Step.
        
        Unlike DDPM, DDIM is deterministic (eta=0) and allows skipping steps.
        It models the reverse process as a non-Markovian process.
        
        Formula:
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + dir_xt * sigma
        where pred_x0 is predicted from x_t and model_output (epsilon).
        
        Args:
            model_output: Predicted noise (epsilon) from the model.
            t: Current timestep (Batch,).
            x_t: Current noisy sample.
            t_prev: Next timestep to jump to (Batch,).
            eta: Stochasticity parameter (0.0 = Deterministic DDIM, 1.0 = DDPM).
            noise_scale: Scaling factor for predicted noise (reduces hiss if < 1.0).
        """
        # Apply Noise Scaling (Dampening)
        # This reduces the amplitude of the predicted noise, effectively smoothing the trajectory
        model_output = model_output * noise_scale

        # 1. Extract coefficients for t
        sqrt_one_minus_at = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_at = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        
        # 2. Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_at * model_output) / sqrt_at
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # 3. Extract coefficients for t_prev
        # Handle t_prev < 0 case (should be 0, but alpha_cumprod[-1] is not defined)
        # We use alphas_cumprod_prev[0] which is 1.0 effectively for t=-1
        # But our arrays are 0-indexed.
        # If t_prev < 0, it means we are at the end. x_prev should be x_0.
        # But let's assume t_prev >= 0.
        
        # Clamp t_prev to be at least 0 for extraction to avoid index out of bounds
        t_prev_clamped = torch.clamp(t_prev, min=0)
        alpha_cumprod_prev = self.extract(self.alphas_cumprod, t_prev_clamped, x_t.shape)
        
        # Special case: if t_prev < 0 (final step), alpha_cumprod_prev should be 1.0
        # We can mask this.
        mask_final = (t_prev < 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        alpha_cumprod_prev = alpha_cumprod_prev * (1 - mask_final) + 1.0 * mask_final
        
        # 4. Compute Variance (sigma)
        # sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
        # For DDIM (eta=0), sigma=0.
        
        sigma = 0.0
        if eta > 0:
            # We don't implement stochastic DDIM fully here for simplicity, 
            # but standard formula can be added if needed.
            pass
            
        # 5. Compute Direction pointing to x_t
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * model_output
        
        # 6. Compute x_prev
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir_xt
        
        return x_prev

    def predict_start_from_noise(self, x_t, t, noise):
        """Helper to reconstruct x_0 for perceptual loss calculation."""
        sqrt_one_minus_at = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_at = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_at * noise) / sqrt_at

class DiffusionUNet(nn.Module):
    """
    Wraps the AudioSuperResNet to add Time Embeddings and Conditioning.
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
        # Time Embedding MLP
        # Maps timestep (scalar) -> embedding vector
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=32),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 32)
        )
        
    def forward(self, x, t, condition):
        """
        Args:
            x: Noisy Target (Batch, 1, Time)
            t: Timesteps (Batch,)
            condition: Low-Res Input (Batch, 1, Time)
        """
        # 1. Generate Time Embedding
        t_emb = self.time_mlp(t)
        
        # 2. Concatenate Condition (Conditional Diffusion)
        # We concatenate along channel dimension.
        # Input to base model becomes (Batch, 2, Time)
        # BUT base model expects (Batch, 1, Time) or we need to change its input layer.
        # The base model 'head' is Conv1d(in_channels, ...).
        # We should update the base model to accept 2 channels if we concatenate.
        # OR we can add them? Concatenation is better.
        # We will assume base_model.head has been updated to accept 2 channels.
        
        model_input = torch.cat([x, condition], dim=1)
        
        # 3. Forward pass through base model, injecting t_emb
        return self.model(model_input, t_emb)
