"""
This file implements an Experience class for storing and managing experience data during reinforcement learning training.

The Experience class provides:
- Flat tensor storage for observations, actions, rewards, etc.
- Array views for faster indexing
- Support for LSTM state storage
- Minibatch handling for training
- CPU/GPU memory management

Key features:
- Stores trajectories in fixed-size tensors
- Supports both CPU and GPU storage with optional CPU offloading
- Handles LSTM hidden states if using recurrent policies
- Provides numpy array views for efficient indexing
- Manages minibatch creation for training
"""

import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

class Experience:
    '''Flat tensor storage and array views for faster indexing'''
    def __init__(self, batch_size, bptt_horizon, minibatch_size, hidden_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
                 cpu_offload=False, device='cuda', lstm=None, lstm_total_agents=0):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload
        self.obs=torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
            pin_memory=pin, device=device if not pin else 'cpu')
        self.actions=torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs=torch.zeros(batch_size, pin_memory=pin)
        self.rewards=torch.zeros(batch_size, pin_memory=pin)
        self.dones=torch.zeros(batch_size, pin_memory=pin)
        self.truncateds=torch.zeros(batch_size, pin_memory=pin)
        self.values=torch.zeros(batch_size, pin_memory=pin)
        self.e3b_inv = 10*torch.eye(hidden_size).repeat(lstm_total_agents, 1, 1).to(device)

        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        assert lstm is not None
        assert lstm_total_agents > 0
        shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
        self.lstm_h = torch.zeros(shape).to(device, non_blocking=True)
        self.lstm_c = torch.zeros(shape).to(device, non_blocking=True)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError(f'batch_size {batch_size} must be divisible by minibatch_size {minibatch_size}')

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError(f'minibatch_size {minibatch_size} must be divisible by bptt_horizon {bptt_horizon}')

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[:self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs.to(self.obs.device, non_blocking=True)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(
            range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(1,0,-1)).to(self.obs.device).long()
        self.b_idxs = self.b_idxs_obs.to(self.device, non_blocking=True)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        return idxs

    def flatten_batch(self, advantages_np):
        advantages = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True, dtype=torch.long)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(self.minibatch_rows,
            self.num_minibatches, self.bptt_horizon).transpose(0, 1).reshape(
            self.num_minibatches, self.minibatch_size)
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values
