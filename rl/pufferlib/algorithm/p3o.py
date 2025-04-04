import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

from .ppo import PPO

class P3O(PPO):
    def __init__(self, horizon):
        self.horizon = horizon

    def make_experience(self, experience, total_agents, batch_size, device):
        super().make_experience(experience, total_agents, batch_size, device)
        experience["values_mean"] = torch.zeros(batch_size, self.horizon, device=device)
        experience["values_std"] = torch.zeros(batch_size, self.horizon, device=device)
        experience["reward_block"] = torch.zeros(batch_size, self.horizon, dtype=torch.float32, device=device)
        experience["mask_block"] = torch.ones(batch_size, self.horizon, dtype=torch.float32, device=device)
        experience["buf"] = torch.zeros(batch_size, self.horizon, dtype=torch.float32, device=device)
        experience["advantages"] = torch.zeros(batch_size, dtype=torch.float32, device=device)
        experience["bounds"] = torch.zeros(batch_size, dtype=torch.int32, device=device)
        experience["vstd_max"] = 1.0

    def initialize_training(self):
        reward_block = experience.reward_block
        mask_block = experience.mask_block
        values_mean = experience.values_mean[idxs]
        values_std = experience.values_std[idxs]
        advantages = experience.advantages

        # Note: This function gets messed up by computing across
        # episode bounds. Because we store experience in a flat buffer,
        # bounds can be crossed even after handling dones. This prevent
        # our method from scaling to longer horizons. TODO: Redo the way
        # we store experience to avoid this issue
        vstd_min = values_std.min().item()
        vstd_max = values_std.max().item()
        torch.cuda.synchronize()

        mask_block.zero_()
        experience.buf.zero_()
        reward_block.zero_()
        r_mean = rewards.mean().item()
        r_std = rewards.std().item()
        advantages.zero_()
        experience.bounds.zero_()

        # TODO: Rename vstd to r_std
        advantages = self.compute_advantages(reward_block, mask_block, values_mean, values_std,
                experience.buf, dones, rewards, advantages, experience.bounds,
                r_std, data.puf, train_cfg.p3o_horizon)

        horizon = torch.where(values_std[0] > 0.95*r_std)[0]
        horizon = horizon[0].item()+1 if len(horizon) else 1
        if horizon < 16:
            horizon = 16

        advantages = advantages.cpu().numpy()
        torch.cuda.synchronize()

        experience.flatten_batch(advantages, reward_block, mask_block)
        torch.cuda.synchronize()

    def store_value(self, experience, dst, value, **kwargs):
        self.values_mean[dst] = value.mean
        self.values_std[dst] = value.std


    def prepare_experience(self, experience, state, idxs):
        super().prepare_experience(experience, state, idxs)
        self.reward_block = torch.as_tensor(reward_block).to(self.device)
        self.b_reward_block = self.reward_block.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon, self.p3o_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size, self.p3o_horizon)

        b_mask_block = torch.as_tensor(mask_block).to(self.device)
        self.b_mask_block = b_mask_block.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon, self.p3o_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size, self.p3o_horizon)

        self.b_values_mean = self.values_mean.to(self.device, non_blocking=True)[b_flat]
        self.b_values_std = self.values_std.to(self.device, non_blocking=True)[b_flat]
        self.b_returns = self.buf.to(self.device, non_blocking=True).reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon, self.p3o_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size, self.p3o_horizon)

    def update_state(self, experience, state):
        state["values_mean"] = experience["b_values_mean"]
        state["values_std"] = experience["b_values_std"]
        state["reward_block"] = experience["b_reward_block"]
        state["mask_block"] = experience["b_mask_block"]

    def compute_value_loss(self, losses):
        newvalue_mean = newvalue.mean.view(-1, train_cfg.p3o_horizon)
        newvalue_std = newvalue.std.view(-1, train_cfg.p3o_horizon)
        newvalue_var = torch.square(newvalue_std)
        criterion = torch.nn.GaussianNLLLoss(reduction='none')
        #v_loss = criterion(newvalue_mean[:, :32], rew_block[:, :32], newvalue_var[:, :32])
        v_loss = criterion(newvalue_mean, rew_block, newvalue_var)
        v_loss = v_loss[:, :(horizon+3)]
        mask_block = mask_block[:, :(horizon+3)]
        #v_loss[:, horizon:] = 0
        #v_loss = (v_loss * mask_block).sum(axis=1)
        #v_loss = (v_loss - v_loss.mean().item()) / (v_loss.std().item() + 1e-8)
        #v_loss = v_loss.mean()
        v_loss = v_loss[mask_block.bool()].mean()
        #TODO: Count mask and sum
        # There is going to have to be some sort of norm here.
        # Right now, learning works at different horizons, but you need
        # to retune hyperparameters. Ideally, horizon should be a stable
        # param that zero-shots the same hypers

        # Faster than masking
        #v_loss = (v_loss*mask_block[:, :32]).sum() / mask_block[:, :32].sum()
        #v_loss = (v_loss*mask_block).sum() / mask_block.sum()
        #v_loss = v_loss[mask_block.bool()].mean()
        losses['value'] = v_loss


