import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNDModule(nn.Module):
    def __init__(self, input_shape, output_size=128):
        super(RNDModule, self).__init__()
        # Flatten the input (works for vector or image observations)
        input_size = int(np.prod(input_shape))
        self.target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        # Freeze the target network parameters
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        target_out = self.target(obs)
        predictor_out = self.predictor(obs)
        return target_out, predictor_out

    def get_intrinsic_reward(self, obs):
        # Compute the L2 error between predictor and target outputs per sample
        with torch.no_grad():
            target_out, predictor_out = self.forward(obs)
        error = (predictor_out - target_out).pow(2).sum(dim=1)
        return error

    def update(self, obs, optimizer):
        # Update predictor network to minimize the squared error
        target_out, predictor_out = self.forward(obs)
        loss = (predictor_out - target_out).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
