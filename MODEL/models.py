from torch import nn


class VAEModel(nn.Module):
    """VAE model

    Args:
        autoencoder: variational autoencoder architecture
    
    Inputs:
        x: treatment variable D or outcome variable Y

    Returns:
        z_mean: mean of posterior distribution of latents (b, z_dim)
        z_log_var: log variance of posterior distribution of latents (b, z_dim)
        target_reconstructed: reconstructed input (b, 1)
    """

    def __init__(self, autoencoder):
        super(VAEModel, self).__init__()
        self.autoencoder = autoencoder

    def forward(self, targets, x = None):
        z_mean, z_log_var, target_reconstructed = self.autoencoder(targets, x)
        return z_mean, z_log_var, target_reconstructed