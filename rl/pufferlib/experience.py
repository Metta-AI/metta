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
    def __init__(self, batch_size, bptt_horizon, minibatch_size, max_minibatch_size, hidden_size,
                 obs_shape, obs_dtype, atn_shape, atn_dtype, cpu_offload=False,
                 device='cuda', lstm=None, lstm_total_agents=0,
                 use_e3b=False, e3b_coef=0.1, e3b_lambda=10.0,
                 use_diayn=False, diayn_archive=128, diayn_coef=0.1,
                 use_p3o=False, p3o_horizon=32):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload
        obs_device = device if not pin else 'cpu'
        self.obs=torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
            pin_memory=pin, device=device if not pin else 'cpu')
        self.actions=torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, device=device)
        self.logprobs=torch.zeros(batch_size, device=device)
        self.rewards=torch.zeros(batch_size, device=device)
        self.dones=torch.zeros(batch_size, device=device)
        self.truncateds=torch.zeros(batch_size, device=device)

        self.use_e3b = use_e3b
        if use_e3b:
            self.e3b_inv = torch.eye(hidden_size).repeat(lstm_total_agents, 1, 1).to(device) / e3b_lambda
            self.e3b_orig = self.e3b_inv.clone()
            self.e3b_mean = None
            self.e3b_std = None

        self.use_diayn = use_diayn
        if use_diayn:
            #self.diayn_archive = torch.randn(diayn_archive, hidden_size, dtype=torch.float32, device=device)
            self.diayn_archive = torch.nn.functional.one_hot(torch.arange(diayn_archive), diayn_archive).to(device).float()
            self.diayn_skills = torch.randint(0, diayn_archive, (lstm_total_agents,), dtype=torch.long, device=device)
            self.diayn_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

        self.use_p3o = use_p3o
        self.p3o_horizon = p3o_horizon
        if use_p3o:
            self.values_mean=torch.zeros(batch_size, p3o_horizon, device=device)
            self.values_std=torch.zeros(batch_size, p3o_horizon, device=device)
            self.reward_block = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.mask_block = torch.ones(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.buf = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self.bounds = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self.vstd_max = 1.0
        else:
            self.values = torch.zeros(batch_size, device=device)

        self.sort_keys = np.zeros((batch_size, 3), dtype=np.int32)
        self.sort_keys[:, 0] = np.arange(batch_size)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        minibatch_size = min(minibatch_size, max_minibatch_size)
        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError('batch_size must be divisible by minibatch_size')

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError('minibatch_size must be divisible by bptt_horizon')

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.p3o_horizon = p3o_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, state, cpu_obs, gpu_obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = np.where(mask)[0]
        num_indices = indices.size
        end = ptr + num_indices
        dst = slice(ptr, end)

        # Zero-copy indexing for contiguous env_id
        if num_indices == mask.size and isinstance(env_id, slice):
            gpu_inds = cpu_inds = slice(0, min(self.batch_size - ptr, num_indices))
        else:
            cpu_inds = indices[:self.batch_size - ptr]
            gpu_inds = torch.as_tensor(cpu_inds).to(self.obs.device, non_blocking=True)

        if self.obs.device.type == 'cuda':
            self.obs[dst] = gpu_obs[gpu_inds]
        else:
            self.obs[dst] = cpu_obs[cpu_inds]

        if self.use_diayn:
            self.diayn_batch[dst] = state.diayn_z_idxs[gpu_inds]

        if self.use_p3o:
            self.values_mean[dst] = value.mean[gpu_inds]
            self.values_std[dst] = value.std[gpu_inds]
        else:
            self.values[dst] = value[gpu_inds].flatten()

        self.actions[dst] = action[gpu_inds]
        self.logprobs[dst] = logprob[gpu_inds]
        self.rewards[dst] = reward[cpu_inds].to(self.rewards.device) # ???
        self.dones[dst] = done[cpu_inds].to(self.dones.device) # ???

        if isinstance(env_id, slice):
            self.sort_keys[dst, 1] = np.arange(env_id.start, env_id.stop, dtype=np.int32)
        else:
            self.sort_keys[dst, 1] = env_id[cpu_inds]

        self.sort_keys[dst, 2] = self.step
        self.ptr = end
        self.step += 1

        return action.cpu().numpy()

    def sort_training_data(self):
        idxs = np.lexsort((self.sort_keys[:, 2], self.sort_keys[:, 1]))
        self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(1,0,-1)).to(self.obs.device).long()
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size)
        self.sort_keys[:, 1:] = 0
        return idxs

    def flatten_batch(self, advantages_np, reward_block=None, mask_block=None):
        advantages = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size)

        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)[b_idxs].contiguous()
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)[b_idxs]
        self.b_dones = self.dones.to(self.device, non_blocking=True)[b_idxs]
        self.b_obs = self.obs[self.b_idxs_obs]

        if self.use_p3o:
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
        else:
            self.b_values = self.values.to(self.device, non_blocking=True)[b_flat]
            self.returns = advantages + self.values # Check sorting of values here
            self.b_returns = self.b_advantages + self.b_values # Check sorting of values here

        if self.use_diayn:
            self.b_diayn_z_idxs = self.diayn_batch.to(self.device, non_blocking=True)[b_flat]
            self.b_diayn_z = self.diayn_archive[self.b_diayn_z_idxs]
