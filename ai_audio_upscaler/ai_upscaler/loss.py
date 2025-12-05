import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTLoss(nn.Module):
    """
    Single-scale STFT Loss module.
    Calculates Spectral Convergence and Log Magnitude Loss.
    """
    def __init__(self, fft_size, hop_size, win_size):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(win_size)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (Batch, Time) or (Batch, Channels, Time)
            target: (Batch, Time) or (Batch, Channels, Time)
        """
        # Ensure inputs are (Batch, Time) for stft
        if input.dim() == 3:
            input = input.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # STFT
        # return_complex=True is required in newer torch versions for stft
        input_stft = torch.stft(input, self.fft_size, self.hop_size, self.win_size, 
                                window=self.window.to(input.device), return_complex=True)
        target_stft = torch.stft(target, self.fft_size, self.hop_size, self.win_size, 
                                 window=self.window.to(target.device), return_complex=True)

        # Magnitude
        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)

        # Spectral Convergence Loss
        sc_loss = torch.norm(target_mag - input_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)

        # Log Magnitude Loss
        log_mag_loss = F.l1_loss(torch.log(target_mag + 1e-7), torch.log(input_mag + 1e-7))

        return sc_loss + log_mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.
    Aggregates STFT losses over multiple resolutions to capture both 
    time and frequency details effectively.
    """
    def __init__(self, 
                 fft_sizes=[1024, 2048, 512], 
                 hop_sizes=[120, 240, 50], 
                 win_sizes=[600, 1200, 240]):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        
        self.stft_losses = nn.ModuleList()
        for fs, hs, ws in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(STFTLoss(fs, hs, ws))

    def forward(self, input, target):
        total_loss = 0.0
        for f in self.stft_losses:
            total_loss += f(input, target)
        return total_loss / len(self.stft_losses)

def feature_loss(fmap_r, fmap_g):
    """
    Feature Matching Loss.
    L1 distance between discriminator feature maps of real and generated audio.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Hinge Loss for Discriminator.
    Minimizes: max(0, 1 - Real) + max(0, 1 + Fake)
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(F.relu(1 - dr))
        g_loss = torch.mean(F.relu(1 + dg))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """
    Hinge Loss for Generator.
    Minimizes: -Fake (i.e., maximize Fake score)
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        loss_val = -torch.mean(dg)
        loss += loss_val
        gen_losses.append(loss_val.item())

    return loss, gen_losses
