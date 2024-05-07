import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

'''
Discriminator(
  (body): Sequential(
    (block1): Sequential(
      (0): Linear(in_features=1536, out_features=1024, bias=True)
      (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2)
    )
  )
  (tail): Linear(in_features=1024, out_features=1, bias=False)
)
'''


'''
ActorCriticCnnPolicy(
  (features_extractor): NatureCNN(
    (cnn): Sequential(
      (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
      (1): ReLU()
      (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
      (5): ReLU()
      (6): Flatten(start_dim=1, end_dim=-1)
    )
    (linear): Sequential(
      (0): Linear(in_features=65536, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (pi_features_extractor): NatureCNN(
    (cnn): Sequential(
      (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
      (1): ReLU()
      (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
      (5): ReLU()
      (6): Flatten(start_dim=1, end_dim=-1)
    )
    (linear): Sequential(
      (0): Linear(in_features=65536, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (vf_features_extractor): NatureCNN(
    (cnn): Sequential(
      (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
      (1): ReLU()
      (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
      (5): ReLU()
      (6): Flatten(start_dim=1, end_dim=-1)
    )
    (linear): Sequential(
      (0): Linear(in_features=65536, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (mlp_extractor): MlpExtractor(
    (shared_net): Sequential()
    (policy_net): Sequential()
    (value_net): Sequential()
  )
  (action_net): Linear(in_features=512, out_features=2, bias=True)
  (value_net): Linear(in_features=512, out_features=1, bias=True)
)
'''
class Discriminator(torch.nn.Module):
    def __init__(self, in_planes=1536, hidden_size=1024, device='cpu'):
        super(Discriminator, self).__init__()
        self.device = device
        self.tensor_length = 2352

        self.body = torch.nn.Sequential(
            nn.Linear(in_features=in_planes, out_features=hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
    # in_planes = 1536  # Example input size
    # hidden_size = 1024
    # discriminator = Discriminator(in_planes, hidden_size)
    # print(discriminator)

    # Create a sample input tensor for testing
    image = torch.randn(17, 3, 256, 256)

    # Forward pass
    discriminator = Discriminator()
    output = discriminator(image)
    print("Output shape:", output.shape)
    print("Output tensor:", output)
