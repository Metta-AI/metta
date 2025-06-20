import inspect
import logging
import sys
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.build_context import BuildContext
from metta.agent.policy_state import PolicyState
from metta.agent.pytorch_policy import PytorchPolicy
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgent(nn.Module):
    """A wrapper for neural network policies that provides a standard interface.

    MettaAgent wraps various policy implementations (BrainPolicy, PyTorch policies, etc.)
    and provides factory methods to create agents from different sources with proper
    build context tracking for reconstruction.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    # Factory methods (class methods)
    @classmethod
    def from_brain_policy(cls, env, cfg: DictConfig) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from BrainPolicy configuration.

        Args:
            env: The environment with observation/action space information
            cfg: Configuration dict containing agent parameters

        Returns:
            Tuple of (MettaAgent, BuildContext) for reconstruction
        """
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Capture environment attributes for reconstruction
        env_attributes = {
            "obs_width": getattr(env, "obs_width", None),
            "obs_height": getattr(env, "obs_height", None),
            "single_observation_space": getattr(env, "single_observation_space", None),
            "single_action_space": getattr(env, "single_action_space", None),
            "feature_normalizations": getattr(env, "feature_normalizations", None),
            "action_names": getattr(env, "action_names", None),
            "max_action_args": getattr(env, "max_action_args", None),
        }

        # Create BrainPolicy
        brain = hydra.utils.instantiate(
            cfg.agent,
            obs_space=obs_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            action_space=env.single_action_space,
            feature_normalizations=env.feature_normalizations,
            device=cfg.device,
            _target_="metta.agent.brain_policy.BrainPolicy",
            _recursive_=False,
        )

        # Capture component source code
        source_code = cls._capture_brain_policy_source_code(brain)

        # Capture the full configuration
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Extract component configurations
        component_configs = {}
        if hasattr(cfg.agent, "components"):
            component_configs = OmegaConf.to_container(cfg.agent.components, resolve=True)

        # Create build context
        build_context = BuildContext(
            method="build_from_brain_policy",
            env_attributes=env_attributes,
            kwargs={"cfg": config_dict},
            source_code=source_code,
            config=config_dict,
            component_configs=component_configs,
        )

        return cls(brain), build_context

    @classmethod
    def from_pytorch_policy(
        cls, path: str, device: str = "cpu", pytorch_cfg: Optional[DictConfig] = None
    ) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from a PyTorch policy checkpoint.

        Args:
            path: Path to the checkpoint file
            device: Device to load the policy on
            pytorch_cfg: Optional configuration for the PyTorch policy

        Returns:
            Tuple of (MettaAgent, BuildContext) for reconstruction
        """
        policy = cls._build_pytorch_policy(path, device, pytorch_cfg)

        # Create build context
        build_context = BuildContext(
            method="build_from_pytorch_policy",
            args=(path,),
            kwargs={"device": device, "pytorch": pytorch_cfg},
        )

        return cls(policy), build_context

    @classmethod
    def from_policy_class(cls, policy_class, *args, **kwargs) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from a custom policy class.

        Args:
            policy_class: The policy class to instantiate
            *args, **kwargs: Arguments to pass to the policy constructor

        Returns:
            Tuple of (MettaAgent, BuildContext) for reconstruction
        """
        policy = policy_class(*args, **kwargs)

        # Ensure it's a PytorchPolicy or wrap it if needed
        if not isinstance(policy, PytorchPolicy):
            logger.warning(f"Policy {policy_class.__name__} does not inherit from PytorchPolicy. Wrapping it.")
            policy = PytorchPolicy(policy)

        # Create build context with class info
        build_context = BuildContext(
            method="build_from_policy_class", args=args, kwargs={**kwargs, "class_name": policy_class.__name__}
        )

        # Capture source code for the policy class
        source_code = cls._capture_policy_source_code(policy)
        build_context.source_code = source_code

        return cls(policy), build_context

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

    @staticmethod
    def _capture_brain_policy_source_code(brain) -> Dict[str, str]:
        """Capture source code for BrainPolicy and all its components."""
        source_code = {}

        # Capture BrainPolicy source
        brain_module = sys.modules.get(brain.__class__.__module__)
        if brain_module and hasattr(brain_module, "__file__") and brain_module.__file__:
            with open(brain_module.__file__, "r") as f:
                source_code["metta.agent.brain_policy"] = f.read()

        # Capture component sources
        if hasattr(brain, "components"):
            for name, component in brain.components.items():
                component_class = component.__class__
                module_name = component_class.__module__

                if module_name in source_code:
                    continue

                module = sys.modules.get(module_name)
                if module and hasattr(module, "__file__") and module.__file__:
                    with open(module.__file__, "r") as f:
                        source_code[module_name] = f.read()

        # Also capture key dependencies
        for module_name in [
            "metta.agent.lib.metta_module",
            "metta.agent.lib.metta_layer",
            "metta.agent.util.distribution_utils",
            "metta.agent.policy_state",
        ]:
            if module_name in source_code:
                continue

            module = sys.modules.get(module_name)
            if module and hasattr(module, "__file__") and module.__file__:
                with open(module.__file__, "r") as f:
                    source_code[module_name] = f.read()

        return source_code

    @staticmethod
    def _capture_policy_source_code(policy: nn.Module) -> Dict[str, str]:
        """Capture source code for a policy class and its dependencies."""
        source_code = {}

        # Handle wrapped policies (e.g., PytorchPolicy wrapping another policy)
        if isinstance(policy, PytorchPolicy) and hasattr(policy, "policy"):
            wrapped_source = MettaAgent._capture_policy_source_code(policy.policy)
            source_code.update(wrapped_source)

        policy_class = policy.__class__
        module_name = policy_class.__module__
        class_name = policy_class.__name__
        full_class_path = f"{module_name}.{class_name}"

        # Save the class source
        try:
            source_code[full_class_path] = inspect.getsource(policy_class)
        except OSError:
            pass  # Built-in classes won't have source

        # Save the module source if possible
        module = sys.modules.get(module_name)
        if module and hasattr(module, "__file__") and module.__file__:
            with open(module.__file__, "r") as f:
                source_code[module_name] = f.read()

        return source_code

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


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig) -> Tuple[MettaAgent, BuildContext]:
    """Create a policy with its build context.

    This is a convenience function for backward compatibility.

    Returns:
        Tuple of (MettaAgent, BuildContext) for reconstruction
    """
    return MettaAgent.from_brain_policy(env, cfg)


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
