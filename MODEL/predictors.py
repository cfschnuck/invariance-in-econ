import torch
import torch.nn as nn

class InvarPredictor(nn.Module):
    def __init__(self, e1_dim):
        super(InvarPredictor, self).__init__()
        self.e1_dim = e1_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.e1_dim, 6),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            # nn.Linear(8, 16),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            nn.Linear(6, 1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x