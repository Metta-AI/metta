import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

class E3B():
    def __init__(self, hidden_size, e3b_lambda, e3b_norm):
        self.encoder = torch.nn.Linear(hidden_size, hidden_size)
        self.e3b_norm = e3b_norm
        self.e3b_lambda = e3b_lambda
        self.hidden_size = hidden_size

    def make_experience(self, experience, lstm_total_agents, batch_size, device):
        experience["inv"] = torch.eye(self.hidden_size).repeat(lstm_total_agents, 1, 1).to(device) / self.e3b_lambda
        experience["orig"] = experience["inv"].clone()
        experience["mean"] = None
        experience["std"] = None


    def on_post_step(self, experience, state):
        e3b = self.experience.e3b_inv[env_id]
        phi = state.hidden.detach()
        u = phi.unsqueeze(1) @ e3b
        b = u @ phi.unsqueeze(2)
        self.experience.e3b_inv[env_id] -= (u.mT @ u) / (1 + b)
        done_inds = env_id[done_mask]
        self.experience.e3b_inv[done_inds] = self.experience.e3b_orig[done_inds]
        e3b_reward = b.squeeze()

        if self.experience.e3b_mean is None:
            self.experience.e3b_mean = e3b_reward.mean()
            self.experience.e3b_std = e3b_reward.std()
        else:
            w = self.e3b_norm
            self.experience.e3b_mean = (1-w)*e3b_reward.mean() + w*self.experience.e3b_mean
            self.experience.e3b_std = (1-w)*e3b_reward.std() + w*self.experience.e3b_std

        return (e3b_reward - self.experience.e3b_mean) / (self.experience.e3b_std + 1e-6)

    def handle_done(self, env_id, done_mask):
        if done_mask.any():
            done_idxs = env_id[done_mask]
            self.experience.e3b_inv[done_idxs] = self.experience.e3b_orig[done_idxs]

