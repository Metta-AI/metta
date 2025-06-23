import logging
from types import SimpleNamespace
from typing import Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.pytorch_policy import PytorchPolicy
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgent(nn.Module):
    """A wrapper for neural network policies that provides a standard interface.

    MettaAgent wraps various policy implementations (ComponentPolicy, PyTorch policies, etc.)
    and provides factory methods to create agents from different sources.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    # Factory methods (class methods)
    @classmethod
    def from_component_policy(cls, env, cfg: DictConfig) -> "MettaAgent":
        """Build a MettaAgent from ComponentPolicy configuration.

        Args:
            env: The environment with observation/action space information
            cfg: Configuration dict containing agent parameters

        Returns:
            MettaAgent instance
        """
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Create ComponentPolicy
        brain = hydra.utils.instantiate(
            cfg.agent,
            obs_space=obs_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            action_space=env.single_action_space,
            feature_normalizations=env.feature_normalizations,
            device=cfg.device,
            _target_="metta.agent.component_policy.ComponentPolicy",
            _recursive_=False,
        )

        return cls(brain)

    # Backward compatibility alias
    from_brain_policy = from_component_policy

    @classmethod
    def from_pytorch_policy(
        cls, path: str, device: str = "cpu", pytorch_cfg: Optional[DictConfig] = None
    ) -> "MettaAgent":
        """Build a MettaAgent from a PyTorch policy checkpoint.

        Args:
            path: Path to the checkpoint file
            device: Device to load the policy on
            pytorch_cfg: Optional configuration for the PyTorch policy

        Returns:
            MettaAgent instance
        """
        policy = cls._build_pytorch_policy(path, device, pytorch_cfg)
        return cls(policy)

    @classmethod
    def from_policy_class(cls, policy_class, *args, **kwargs) -> "MettaAgent":
        """Build a MettaAgent from a custom policy class.

        Args:
            policy_class: The policy class to instantiate
            *args, **kwargs: Arguments to pass to the policy constructor

        Returns:
            MettaAgent instance
        """
        policy = policy_class(*args, **kwargs)

        # Ensure it's a PytorchPolicy or wrap it if needed
        if not isinstance(policy, PytorchPolicy):
            logger.warning(f"Policy {policy_class.__name__} does not inherit from PytorchPolicy. Wrapping it.")
            policy = PytorchPolicy(policy)

        return cls(policy)

    @staticmethod
    def _build_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: Optional[DictConfig] = None):
        """Build a policy from a PyTorch checkpoint.

        This method creates a policy from PyTorch checkpoint weights by instantiating
        it with the proper configuration and wrapping it for use with MettaAgent.
        """
        weights = torch.load(path, map_location=device, weights_only=True)

        try:
            num_actions, hidden_size = weights["policy.actor.0.weight"].shape
            num_action_args, _ = weights["policy.actor.1.weight"].shape
            _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
        except Exception as e:
            logger.warning(f"Failed automatic parse from weights: {e}")
            # TODO -- fix all magic numbers
            num_actions, num_action_args = 9, 10
            _, obs_channels = 128, 34

        # Create environment namespace
        env = SimpleNamespace(
            single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
            single_observation_space=SimpleNamespace(shape=tuple(torch.tensor([obs_channels, 11, 11]).tolist())),
        )

        policy = instantiate(pytorch_cfg, env=env, policy=None)
        policy.load_state_dict(weights)

        wrapped_policy = PytorchPolicy(policy)
        wrapped_policy.hidden_size = hidden_size
        return wrapped_policy.to(device)

    # Instance methods
    def __getattr__(self, name):
        """Pass through attributes to the underlying policy."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.forward(x, state, action)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        else:
            logger.warning(
                f"Policy of type {type(self.policy).__name__} does not have 'activate_actions' method. Skipping."
            )

    @property
    def lstm(self):
        return self.policy.lstm

    @property
    def total_params(self):
        return sum(p.numel() for p in self.policy.parameters())

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete action indices."""
        return self.policy._convert_action_to_logit_index(flattened_action)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.policy._convert_logit_index_to_action(action_logit_index)

    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """Apply a method to all components."""
        return getattr(self.policy, "_apply_to_components", lambda *a, **k: [])(method_name, *args, **kwargs)

    def _delegate_if_exists(self, method_name, *args, **kwargs):
        """Helper to delegate method calls to wrapped policy if the method exists."""
        if hasattr(self.policy, method_name):
            return getattr(self.policy, method_name)(*args, **kwargs)

    def _get_loss_or_zero(self, loss_name: str) -> torch.Tensor:
        """Helper to get a loss value from the policy or return zero tensor."""
        if hasattr(self.policy, loss_name):
            return getattr(self.policy, loss_name)()
        return torch.tensor(0.0, device=getattr(self, "device", "cpu"))

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss."""
        return self._get_loss_or_zero("l2_reg_loss")

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss."""
        return self._get_loss_or_zero("l2_init_loss")

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval."""
        self._delegate_if_exists("update_l2_init_weight_copy")

    def clip_weights(self):
        """Weight clipping."""
        self._delegate_if_exists("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for all components."""
        return self._delegate_if_exists("compute_weight_metrics", delta) or []


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig) -> MettaAgent:
    """Create a policy.

    This is a convenience function for backward compatibility.

    Returns:
        MettaAgent instance
    """
    return MettaAgent.from_component_policy(env, cfg)


def make_distributed_agent(agent: Union[MettaAgent, nn.Module], device: torch.device) -> DistributedDataParallel:
    """Create a distributed version of an agent for multi-GPU training.

    Args:
        agent: The agent to distribute
        device: The device to use

    Returns:
        A DistributedDataParallel wrapper around the agent
    """
    logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
    agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
    return DistributedMettaAgent(agent, device)


class DistributedMettaAgent(DistributedDataParallel):
    """A distributed wrapper for MettaAgent that preserves the MettaAgent interface."""

    def __init__(self, agent, device):
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)
