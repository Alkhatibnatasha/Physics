import torch
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(886, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x