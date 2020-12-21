## torch
import torch
import torch.nn as nn

class Disentangler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Disentangler, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.in_dim), 
            nn.Linear(self.in_dim, 16),
            nn.Dropout(p=0.3), 
            nn.LeakyReLU(),
            nn.Linear(16, self.out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x