import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

from .ppo import PPO

class DIAYN():
    def __init__(self, archive_size, loss_coef):
        self.hidden_size = 128
        self.archive_size = archive_size
        self.discriminator = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
                torch.nn.ReLU(),
                pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, archive_size))
            )

        # Right now, I'm just adding the output of this to the hidden state because I don't know how
        # to modify your policy setup. You'd ideally want to concat this with the rest of your embeddings
        self.encoder = torch.nn.Linear(archive_size, self.hidden_size)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def make_experience(self, experience, lstm_total_agents, batch_size, device):
        experience["diayn_archive"] = torch.nn.functional.one_hot(torch.arange(self.archive_size), self.archive_size).to(device).float()
        experience["skills"] = torch.randint(0, self.archive_size, (lstm_total_agents,), dtype=torch.long, device=device)
        experience["batch"] = torch.zeros(batch_size, dtype=torch.long, device=device)


    def update_state(self, state, experience_idxs):
        z_idxs = self.skills[experience_idxs]
        state.diayn_z = self.archive[z_idxs]
        state.diayn_z_idxs = z_idxs

    def compute_reward(self, state):
        q = self.discriminator(state.hidden).squeeze()
        r_diayn = torch.log_softmax(q, dim=-1).gather(-1, state.diayn_z_idxs.unsqueeze(-1)).squeeze()
        return r_diayn # - np.log(1/data.diayn_archive)

    def handle_done(self, env_id, done_mask):
        idxs = env_id[done_mask]
        if len(idxs) > 0:
            z_idxs = torch.randint(0, self.archive_size, (done_mask.sum(),)).to(self.device)
            self.skills[idxs] = z_idxs

    def compute_loss(self, state, z_idxs, losses):
        discriminator = self.policy.diayn_discriminator
        q = discriminator(state.hidden).squeeze()
        diayn_loss = self.cross_entropy(q, z_idxs)
        losses['diayn_loss'] = self.loss_coef * diayn_loss
        torch.cuda.synchronize()

    def store_experience(self, experience, dst, state, **kwargs):
        experience[dst] = state["diayn_z_idxs"]

    def initialize_training(self, experience, state, b_flat):
        experience["b_diayn_z_idxs"] = self.batch.to(self.device, non_blocking=True)[b_flat]
        experience["b_diayn_z"] = self.archive[experience["b_diayn_z_idxs"]]

