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
            nn.BatchNorm1d(self.in_dim),
            nn.Linear(self.in_dim, 16),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16), 
            nn.Linear(16, self.e1_dim + self.e2_dim),
            nn.Dropout(p=0.3), 
            nn.LeakyReLU(),
        )
        self.fc_e1 = nn.Sequential(
            nn.BatchNorm1d(self.e1_dim + self.e2_dim),
            nn.Linear(self.e1_dim + self.e2_dim, self.e1_dim),
            nn.Dropout(p=0.1),
            nn.Tanh(),
        )
        self.fc_e2 = nn.Sequential(
            nn.BatchNorm1d(self.e1_dim + self.e2_dim),
            nn.Linear(self.e1_dim + self.e2_dim, self.e2_dim),
            nn.Dropout(p=0.1),
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
            nn.BatchNorm1d(self.e1_dim + self.e2_dim),
            nn.Linear(self.e1_dim + self.e2_dim, 16),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, self.out_dim),
        )

    def forward(self, e1, e2):
        return self.fc(torch.cat((e1, e2), axis=-1))


# use VAE
class VAELinearAutoencoder(nn.Module):
    def __init__(self, in_dim, z_dim, x_dim):
        super(LinearVAEConceptizer, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.encoder = VAELinearEncoder(self.in_dim, self.z_dim)
        self.decoder = VAELinearDecoder(self.z_dim + self.x_dim, self.in_dim)
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        sample = self.sample(mean, log_var)
        decoded = self.decoder(sample)
        return mean, log_var, decoded.view_as(x)
        
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
    def sample(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z

class VAELinearEncoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(VAELinearEncoder, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.in_dim),
            nn.Linear(self.in_dim, 16),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16), 
            nn.Linear(16, self.z_dim),
            nn.Dropout(p=0.3), 
            nn.LeakyReLU(),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(8, z_dim),
            nn.Tanh()
        )  
        self.log_var_layer = nn.Linear(8, z_dim)

        def forward(self, x):
            encoded = self.fc(x)
            mean = self.mean_layer(encoded)
            log_var = self.log_var_layer(encoded)
            return mean, log_var

class VAELinearDecoder(nn.Module):
    def __init__(self, z_dim, out_dim):
        super(VAELinearDecoder, self).__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        self.fc = nn.Sequential(  
            nn.BatchNorm1d(self.z_dim),
            nn.Linear(self.z_dim, 16),
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
        return decoded