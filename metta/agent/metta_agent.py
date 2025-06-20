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

        This loads a policy using the build_pytorch_policy function which handles
        legacy policies and wraps them appropriately.
        """
        from metta.agent.policy_store import build_pytorch_policy

        policy = build_pytorch_policy(path, self._cfg.device, pytorch=self._cfg.get("pytorch", None))
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

    def build_from_source_file(self, path: str) -> "MettaAgent":
        """Build a MettaAgent by reconstructing it from a saved source file."""
        checkpoint = torch.load(path, map_location=self._cfg.device)

        # Get class path and source code
        model_class_path = checkpoint.get("model_class_path")
        source_code = checkpoint.get("source_code", {})

        if not model_class_path or not source_code:
            raise ValueError("File does not contain source code for reconstruction")

        # Reconstruct the model
        try:
            import importlib.util

            module_name, class_name = model_class_path.rsplit(".", 1)

            # Load the class from the saved source code
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)

            # Use the module's source if available, otherwise just the class source
            module_source = source_code.get(module_name, source_code.get(model_class_path))
            exec(module_source, module.__dict__)

            model_class = getattr(module, class_name)

            # This assumes a default constructor, may need to pass args
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])

            logger.info(f"Reconstructed model from class {model_class_path}")
            return MettaAgent(model)

        except Exception as e:
            logger.error(f"Failed to reconstruct model: {e}")
            raise ValueError(f"Could not reconstruct model from {path}")


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
