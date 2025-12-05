import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseInjection(nn.Module):
    """
    Injects random noise into the feature map, scaled by a learnable factor.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        # Generate noise of the same shape as x
        # This stochasticity helps model high-frequency details that are inherently unpredictable
        noise = torch.randn_like(x, device=x.device)
        return x + self.weight * noise

class GatedResidualBlock(nn.Module):
    """
    Residual Block with Gated Activation Units (WaveNet style) and Noise Injection.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, time_emb_dim=None):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        
        # Gated Activation
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        
        self.noise1 = NoiseInjection(channels)
        self.noise2 = NoiseInjection(channels)
        
        self.skip_conv = nn.Conv1d(channels, channels, 1) if channels != channels else nn.Identity()
        
        # Time Embedding Projection
        self.time_proj = nn.Linear(time_emb_dim, channels) if time_emb_dim else None

    def forward(self, x, time_emb=None):
        residual = x
        
        # Layer 1: Dilated Conv -> Noise -> Gated Activation
        c = self.conv1(x)
        
        # Inject Time Embedding
        if self.time_proj is not None and time_emb is not None:
            # Project time embedding to channels and unsqueeze for time dimension
            # time_emb: (Batch, time_emb_dim) -> (Batch, channels) -> (Batch, channels, 1)
            t = self.time_proj(time_emb).unsqueeze(-1)
            c = c + t
            
        c = self.noise1(c)
        
        # Gated Activation (WaveNet style)
        # Tanh acts as the "signal" filter (range -1 to 1)
        # Sigmoid acts as the "gate" (range 0 to 1), controlling information flow
        f = self.conv_filter(c)
        g = self.conv_gate(c)
        act = torch.tanh(f) * torch.sigmoid(g)
        
        # Layer 2: Conv -> Noise
        out = self.conv2(act)
        out = self.noise2(out)
        
        # Residual Connection
        return residual + out

class AudioSuperResNet(nn.Module):
    """
    Generative Bandwidth Extension Network (BWE-UNet).
    
    Architecture Overview:
    - **Backbone**: U-Net style Encoder-Decoder with skip connections.
    - **Building Block**: Gated Residual Blocks (inspired by WaveNet) with dilated convolutions.
    - **Stochasticity**: Noise Injection layers allow the model to hallucinate plausible high-frequency content.
    - **Global Residual**: The network learns the *difference* (residual) between the upsampled baseline and the target high-res audio.
    
    Input: (Batch, 1, Time) - Baseline upsampled audio (e.g., via Sinc interpolation)
    Output: (Batch, 1, Time) - Refined high-resolution audio
    """
    def __init__(self, in_channels=1, base_channels=32, num_layers=4, time_emb_dim=None):
        super().__init__()
        
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        self.head = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        curr_channels = base_channels
        
        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(
                    GatedResidualBlock(curr_channels, dilation=1, time_emb_dim=time_emb_dim),
                    GatedResidualBlock(curr_channels, dilation=2, time_emb_dim=time_emb_dim),
                )
            )
            self.downsamples.append(nn.Conv1d(curr_channels, curr_channels * 2, kernel_size=4, stride=2, padding=1))
            curr_channels *= 2
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            GatedResidualBlock(curr_channels, dilation=1, time_emb_dim=time_emb_dim),
            GatedResidualBlock(curr_channels, dilation=2, time_emb_dim=time_emb_dim),
            GatedResidualBlock(curr_channels, dilation=4, time_emb_dim=time_emb_dim),
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(num_layers):
            self.upsamples.append(nn.ConvTranspose1d(curr_channels, curr_channels // 2, kernel_size=4, stride=2, padding=1))
            curr_channels //= 2
            self.decoders.append(
                nn.Sequential(
                    GatedResidualBlock(curr_channels, dilation=1, time_emb_dim=time_emb_dim),
                    GatedResidualBlock(curr_channels, dilation=2, time_emb_dim=time_emb_dim),
                )
            )
            
        self.tail = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3) # Output is always 1 channel (Audio)
        
    def forward(self, x, time_emb=None):
        """
        Args:
            x: Input tensor (Batch, in_channels, Time)
            time_emb: Optional time embedding (Batch, time_emb_dim)
        Returns:
            Refined tensor (Batch, 1, Time)
        """
        # Save input for global residual
        # Logic:
        # - GAN Mode (time_emb is None): We predict the residual (difference) to add to the input.
        #   Output = Input + Model(Input)
        # - Diffusion Mode (time_emb is present): We predict the noise (epsilon) to SUBTRACT.
        #   Output = Model(Input, t) -> Epsilon
        #   The diffusion sampler handles the subtraction.
        
        use_residual = (time_emb is None)
        
        if use_residual:
            residual_global = x
        
        # Head
        x = self.head(x)
        
        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            # Pass time_emb to blocks
            for layer in enc:
                if isinstance(layer, GatedResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)
            skips.append(x)
            x = down(x)
            
        # Bottleneck
        for layer in self.bottleneck:
             if isinstance(layer, GatedResidualBlock):
                x = layer(x, time_emb)
             else:
                x = layer(x)
        
        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            # Skip connection
            skip = skips[-(i+1)]
            
            # Handle size mismatch (center crop/pad)
            if x.shape[-1] != skip.shape[-1]:
                # Crop or pad x to match skip
                diff = x.shape[-1] - skip.shape[-1]
                if diff > 0:
                    # Crop x (center)
                    start = diff // 2
                    end = start + skip.shape[-1]
                    x = x[..., start:end]
                elif diff < 0:
                    # Pad x
                    pad_left = abs(diff) // 2
                    pad_right = abs(diff) - pad_left
                    x = F.pad(x, (pad_left, pad_right))
            
            x = x + skip 
            
            for layer in dec:
                if isinstance(layer, GatedResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)
            
        # Tail
        out = self.tail(x)
        
        # Global Residual
        if use_residual:
            return residual_global + out
        else:
            return out
