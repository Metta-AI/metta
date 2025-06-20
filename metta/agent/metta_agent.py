import logging
from typing import Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgentBuilder:
    def __init__(self, cfg):
        self._cfg = cfg

    def build_from_brain_policy(self, env) -> "MettaAgent":
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Here's where we create a BrainPolicy.
        brain = hydra.utils.instantiate(
            self._cfg.agent,
            obs_space=obs_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            action_space=env.single_action_space,
            feature_normalizations=env.feature_normalizations,
            device=self._cfg.device,
            _target_="metta.agent.brain_policy.BrainPolicy",
            _recursive_=False,
        )
        return MettaAgent(brain)

    def build_from_pytorch_policy(self, path) -> "MettaAgent":
        """Build a MettaAgent from a PyTorch policy checkpoint.

        This loads a policy using the load_policy function which handles
        legacy policies and wraps them appropriately.
        """
        from metta.rl.policy import load_policy

        policy = load_policy(path, self._cfg.device, puffer=self._cfg.puffer)
        return MettaAgent(policy)

    def build_from_policy_class(self, policy_class, *args, **kwargs) -> "MettaAgent":
        """Build a MettaAgent from a custom policy class.

        This allows external policies to be directly instantiated and wrapped.

        Args:
            policy_class: The policy class to instantiate (should inherit from PytorchPolicy)
            *args, **kwargs: Arguments to pass to the policy constructor
        """
        from metta.agent.pytorch_policy import PytorchPolicy

        policy = policy_class(*args, **kwargs)

        # Ensure it's a PytorchPolicy or wrap it if needed
        if not isinstance(policy, PytorchPolicy):
            logger.warning(f"Policy {policy_class.__name__} does not inherit from PytorchPolicy. Wrapping it.")
            policy = PytorchPolicy(policy)

        return MettaAgent(policy)

    def load_from_disk(self, path) -> "MettaAgent":
        """Load a model from disk, supporting torch.jit models."""
        try:
            # Try loading as torch.jit model first
            model = torch.jit.load(path, map_location=self._cfg.device)
            logger.info(f"Successfully loaded torch.jit model from {path}")
            return MettaAgent(model)
        except Exception as e:
            logger.debug(f"Not a torch.jit model ({e}), falling back to PolicyStore")
            # Fall back to PolicyStore loading for regular checkpoints
            raise NotImplementedError(
                "Loading non-jit models should go through PolicyStore. Use PolicyStore.load_from_uri instead."
            )


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig) -> "MettaAgent":
    builder = MettaAgentBuilder(cfg)
    return builder.build_from_brain_policy(env)


class DistributedMettaAgent(DistributedDataParallel):
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


class MettaAgent(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

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
        """
        Convert (action_type, action_param) pairs to discrete action indices
        using precomputed offsets.

        Args:
            flattened_action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs

        Returns:
            action_logit_indices: Tensor of shape [B*T] containing flattened action indices
        """
        # Delegate to the wrapped policy
        return self.policy._convert_action_to_logit_index(flattened_action)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs using tensor indexing.

        Args:
            action_logit_index: Tensor of shape [B*T] containing flattened action indices

        Returns:
            action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs
        """
        # Delegate to the wrapped policy
        return self.policy._convert_logit_index_to_action(action_logit_index)

    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """
        Apply a method to all components, collecting and returning the results.

        Args:
            method_name: Name of the method to call on each component
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: Results from calling the method on each component

        Raises:
            AttributeError: If any component doesn't have the requested method
            TypeError: If a component's method is not callable
            AssertionError: If no components are available
        """
        # Delegate to the wrapped policy if it has this method
        if hasattr(self.policy, "_apply_to_components"):
            return self.policy._apply_to_components(method_name, *args, **kwargs)

        # Otherwise, for non-BrainPolicy policies, return empty list
        return []

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust
        it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        if hasattr(self.policy, "l2_reg_loss"):
            return self.policy.l2_reg_loss()
        else:
            return torch.tensor(0.0, device=getattr(self, "device", "cpu"))

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        else:
            return torch.tensor(0.0, device=getattr(self, "device", "cpu"))

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval. 0 means no updating."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def clip_weights(self):
        """Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value
        effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global
        loss value or 0 to turn it off."""
        if hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for all components that have weights enabled for analysis.
        Returns a list of metric dictionaries, one per component. Set analyze_weights to True in the config to turn it
        on for a given component."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []

    def save_jit(self, path: str, example_input=None):
        """Save the model as torch.jit for version-independent loading.

        Args:
            path: Path to save the jit model
            example_input: Optional example input for tracing. If not provided,
                          will use scripting instead of tracing.
        """
        try:
            if example_input is not None:
                # Use tracing with example input
                traced = torch.jit.trace(self.policy, example_input)
                torch.jit.save(traced, path)
                logger.info(f"Saved traced torch.jit model to {path}")
            else:
                # Use scripting
                scripted = torch.jit.script(self.policy)
                torch.jit.save(scripted, path)
                logger.info(f"Saved scripted torch.jit model to {path}")
        except Exception as e:
            logger.error(f"Failed to save as torch.jit: {e}")
            raise
