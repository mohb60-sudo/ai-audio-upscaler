import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorS(nn.Module):
    """
    Spectral Discriminator.
    Operates on a spectrogram with specific resolution.
    """
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator (MRD).
    Consists of multiple DiscriminatorS instances operating on STFTs with different parameters.
    """
    def __init__(self, resolutions=None):
        super(MultiResolutionDiscriminator, self).__init__()
        if resolutions is None:
            # Format: (n_fft, hop_length, win_length)
            # These are standard high-fidelity audio resolutions
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
            
        self.resolutions = resolutions
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True) for _ in resolutions])

    def forward(self, y, y_hat):
        """
        Args:
            y: Real audio (Batch, 1, Time)
            y_hat: Generated audio (Batch, 1, Time)
        Returns:
            y_disc_r: List of scores for real audio
            y_disc_g: List of scores for generated audio
            fmap_r: List of feature maps for real audio
            fmap_g: List of feature maps for generated audio
        """
        y_disc_r, y_disc_g = [], []
        fmap_r, fmap_g = [], []

        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            # Compute Spectrograms
            # We use magnitude spectrograms
            y_spec = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, 
                                window=torch.hann_window(win).to(y.device), 
                                return_complex=True)
            y_spec = torch.abs(y_spec).unsqueeze(1) # (Batch, 1, Freq, Time)
            
            y_hat_spec = torch.stft(y_hat.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, 
                                    window=torch.hann_window(win).to(y.device), 
                                    return_complex=True)
            y_hat_spec = torch.abs(y_hat_spec).unsqueeze(1)

            # Pass through sub-discriminator
            score_r, f_r = self.discriminators[i](y_spec)
            score_g, f_g = self.discriminators[i](y_hat_spec)
            
            y_disc_r.append(score_r)
            y_disc_g.append(score_g)
            fmap_r.append(f_r)
            fmap_g.append(f_g)

        return y_disc_r, y_disc_g, fmap_r, fmap_g
