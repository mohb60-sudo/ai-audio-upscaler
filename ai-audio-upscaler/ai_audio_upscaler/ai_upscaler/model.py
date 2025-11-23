import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Simple 1D Residual Block.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class AudioSuperResNet(nn.Module):
    """
    Lightweight 1D CNN for Audio Super Resolution / Bandwidth Extension.
    
    Takes an upsampled (interpolated) waveform and predicts a residual correction
    to recover high-frequency details.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        
        # Initial feature extraction
        self.head = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)
        
        # Residual backbone
        self.body = nn.Sequential(*[
            ResidualBlock(hidden_channels, kernel_size=3, dilation=2**i) 
            for i in range(num_blocks)
        ])
        
        # Reconstruction
        self.tail = nn.Conv1d(hidden_channels, in_channels, kernel_size=7, padding=3)
        
        # Global skip connection is handled in forward (learning residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Time).
               This should be the baseline-upsampled audio.
        
        Returns:
            Refined audio tensor of same shape.
        """
        residual = x
        
        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)
        
        # Add predicted high-freq residual to the input (Global Skip Connection)
        return residual + out
