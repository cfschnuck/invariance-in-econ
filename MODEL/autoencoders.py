from abc import abstractmethod
import warnings

## torch
import torch
import torch.nn as nn

## autoencoder meta-class
class InvarAutoencoder(nn.Module):
    def __init__(self, in_dim, e1_dim, e2_dim, out_dim, p_noise = 0.5):       
        super(InvarAutoencoder, self).__init__()
        self.in_dim = in_dim
        self.e1_dim = e1_dim
        self.e2_dim = e2_dim
        self.out_dim = out_dim
        self.noise = nn.Dropout(p_noise)

    def forward(self, x):       
        e1, e2 = self.encode(x)
        e1_noisy = self.noise(e1)
        decoded = self.decode(e1_noisy, e2)
        return e1, e2, decoded

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

class LinearAutoencoder(InvarAutoencoder):
    def __init__(self, in_dim, e1_dim, e2_dim, out_dim):
        super(LinearAutoencoder, self).__init__(in_dim, e1_dim, e2_dim, out_dim)
        self.encoder = LinearEncoder(self.in_dim, self.e1_dim, self.e2_dim)
        self.decoder = LinearDecoder(self.e1_dim, self.e2_dim, self.out_dim)

    def encode(self, x):
        e1, e2 = self.encoder(x)
        return e1, e2
    
    def decode(self, e1_noisy, e2):
        return self.decoder(e1_noisy, e2)

class LinearEncoder(nn.Module):
    def __init__(self, in_dim, e1_dim, e2_dim):
        super(LinearEncoder, self).__init__()
        self.in_dim = in_dim
        self.e1_dim = e1_dim
        self.e2_dim = e2_dim
        self.fc = nn.Sequential(
            # nn.Linear(self.in_dim, 6),
            # nn.ReLU(), 
            nn.Linear(self.in_dim, self.e1_dim + self.e2_dim),
            nn.LeakyReLU(),
        )
        self.fc_e1 = nn.Sequential(
            nn.Linear(self.e1_dim + self.e2_dim, self.e1_dim),
            nn.Tanh(),
        )
        self.fc_e2 = nn.Sequential(
            nn.Linear(self.e1_dim + self.e2_dim, self.e2_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        out = self.fc(x)
        e1 = self.fc_e1(out)
        e2 = self.fc_e2(out)
        return e1, e2

# decodes latent space to X
class LinearDecoder(nn.Module):
    def __init__(self, e1_dim, e2_dim, out_dim):
        super(LinearDecoder, self).__init__()
        self.e1_dim = e1_dim
        self.e2_dim = e2_dim
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(self.e1_dim + self.e2_dim, 10),
            nn.LeakyReLU(),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            nn.Linear(10, self.out_dim)
        )

    def forward(self, e1, e2):
        return self.fc(torch.cat((e1, e2), axis=-1))
