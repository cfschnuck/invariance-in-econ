import torch
import torch.nn as nn

class InvarPredictor(nn.Module):
    def __init__(self, e1_dim):
        super(InvarPredictor, self).__init__()
        self.e1_dim = e1_dim
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.e1_dim),
            nn.Linear(self.e1_dim, 8),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            nn.BatchNorm1d(8), 
            nn.Linear(8, 1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x