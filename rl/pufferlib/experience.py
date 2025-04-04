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

from tensordict import TensorDict
import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

class Experience:
    '''Flat tensor storage and array views for faster indexing'''
    def __init__(self,
                 batch_size,
                 bptt_horizon,
                 minibatch_size,
                 max_minibatch_size,
                 device):

        self.data = TensorDict(
            device=device
        )

        if minibatch_size is None:
            minibatch_size = batch_size

        self.sort_keys = np.zeros((batch_size, 3), dtype=np.int32)
        self.sort_keys[:, 0] = np.arange(batch_size)

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
        self.minibatch_size = minibatch_size
        self.device = device
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

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



