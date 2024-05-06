import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

'''
Discriminator(
  (body):
  (block1): Sequential(
    (0): Linear(in_features=1536, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (tail): Linear(in_features=7, out_features=1, bias=False)
)
'''

class Discriminator(torch.nn.Module):
    def __init__(self, in_planes=4, hidden_size=1024, device='cpu'):
        super(Discriminator, self).__init__()
        self.device = device
        self.body = torch.nn.Sequential(
            nn.Linear(in_features=in_planes, out_features=hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.tail = torch.nn.Linear(hidden_size, out_features=2, bias=False)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x

    def sample_action(self, obs, epsilon):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(self.device)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

if __name__ == '__main__':
    # Initialize the discriminator
    in_planes = 1536  # Example input size
    hidden_size = 1024
    discriminator = Discriminator(in_planes, hidden_size)
    print(discriminator)

    # Create a sample input tensor for testing
    sample_input = torch.randn(3, in_planes)  # Assuming batch size 1

    # Forward pass
    output = discriminator(sample_input)
    print("Output shape:", output.shape)
    print("Output tensor:", output)
