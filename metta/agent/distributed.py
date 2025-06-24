"""Distributed training wrapper for Metta agents."""

import logging

import torch
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger("metta.agent.distributed")


class DistributedMettaAgent(DistributedDataParallel):
    """Wrapper for distributed training of agents."""

    def __init__(self, agent, device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
        agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)

    @property
    def lstm(self):
        """Access LSTM from the wrapped module."""
        return self.module.lstm

    @property
    def total_params(self):
        """Access total_params from the wrapped module."""
        return self.module.total_params

    def l2_reg_loss(self):
        """Access l2_reg_loss from the wrapped module."""
        return self.module.l2_reg_loss()

    def l2_init_loss(self):
        """Access l2_init_loss from the wrapped module."""
        return self.module.l2_init_loss()

    def update_l2_init_weight_copy(self):
        """Access update_l2_init_weight_copy from the wrapped module."""
        self.module.update_l2_init_weight_copy()

    def clip_weights(self):
        """Access clip_weights from the wrapped module."""
        self.module.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01):
        """Access compute_weight_metrics from the wrapped module."""
        return self.module.compute_weight_metrics(delta)
