from abc import abstractmethod
import warnings

## torch
import torch
import torch.nn as nn



class VAELinearAutoencoder(nn.Module):
    """Variational autoencoder architecture module

    Args:
        in_dim: dimension of input (treatment variable D or outcome variable Y)
        z_dim: number of latent variables / dimension of latent representation
        x_dim: dimension of confounder X
        target: indicator whether data is treatment variable D or outcome variable Y

    Returns:
        VAE module

    """
    def __init__(self, in_dim, z_dim, x_dim, target):
        super(VAELinearAutoencoder, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.target = target
        self.encoder = VAEEncoder(self.in_dim, self.z_dim)
        self.decoder = VAEDecoder(self.z_dim + self.x_dim, self.in_dim, target)
    
    def forward(self, target, x):
        mean, log_var = self.encoder(target)
        sample = self.sample(mean, log_var)
        # for training
        decoded = self.decoder(torch.cat((sample, x), axis=-1))
        return mean, log_var, decoded
        
    
    def sample(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z

class VAEEncoder(nn.Module):
    """VAE Encoder

    Args:
        in_dim: dimension of input (treatment variable D or outcome variable Y)
        z_dim: number of latent variables / dimension of latent representation
    
    Input:
        x: input (b, in_dim)
    
    Output:
        latent vector (b, z_dim)

    """
    def __init__(self, in_dim, z_dim):
        super(VAEEncoder, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, 16),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16), 
            nn.Linear(16, 16),
            nn.Dropout(p=0.3), 
            nn.LeakyReLU(),
        )
        self.mean_layer = nn.Linear(16, z_dim)
        self.log_var_layer = nn.Linear(16, z_dim)

    def forward(self, x):
        encoded = self.fc(x)
        mean = self.mean_layer(encoded)
        log_var = self.log_var_layer(encoded)
        return mean, log_var

class VAEDecoder(nn.Module):
    """VAE Decoder

    Args:
        zc_dim: dimension of latent represention + dimension of confounder x
        out_dim: dimension of input (treatment variable D or outcome variable Y)
    
    Input:
        x: input (b, zc_dim)
        the decoder receives a concatetation of the latent variables and the confounder X 
    
    Output:
        reconstruction (b, out_dim)

    """
    def __init__(self, zc_dim, out_dim, target):
        super(VAEDecoder, self).__init__()
        self.zc_dim = zc_dim
        self.out_dim = out_dim
        self.target = target
        self.fc = nn.Sequential(  
            nn.Linear(self.zc_dim, 16, bias = False),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, self.out_dim),
        )
    
    def forward(self, z):
        decoded = self.fc(z)
        # add sigmoid activation function to predict D
        if self.target == "D":
            decoded = torch.sigmoid(decoded)
        return decoded