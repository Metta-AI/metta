import logging
from typing import Dict, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.build_context import BuildContext
from metta.agent.policy_state import PolicyState
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgentBuilder:
    def __init__(self, cfg):
        self._cfg = cfg

    def build_from_brain_policy(self, env) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from BrainPolicy configuration.

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

        # Capture component source code
        source_code = self._capture_brain_policy_source_code(brain)

        # Capture the full configuration
        from omegaconf import OmegaConf

        config_dict = OmegaConf.to_container(self._cfg, resolve=True)

        # Extract component configurations
        component_configs = {}
        if hasattr(self._cfg.agent, "components"):
            component_configs = OmegaConf.to_container(self._cfg.agent.components, resolve=True)

        # Create build context
        build_context = BuildContext(
            method="build_from_brain_policy",
            env_attributes=env_attributes,
            kwargs={"cfg": config_dict},  # Store full config for reconstruction
            source_code=source_code,
            config=config_dict,
            component_configs=component_configs,
        )

        return MettaAgent(brain), build_context

    def _capture_brain_policy_source_code(self, brain) -> Dict[str, str]:
        """Capture source code for BrainPolicy and all its components."""
        import sys

        source_code = {}

        # Capture BrainPolicy source
        try:
            brain_module = sys.modules[brain.__class__.__module__]
            if hasattr(brain_module, "__file__") and brain_module.__file__:
                with open(brain_module.__file__, "r") as f:
                    source_code["metta.agent.brain_policy"] = f.read()
        except Exception as e:
            logger.warning(f"Could not capture BrainPolicy source: {e}")

        # Capture component sources
        if hasattr(brain, "components"):
            for name, component in brain.components.items():
                try:
                    component_class = component.__class__
                    module_name = component_class.__module__

                    # Skip if already captured
                    if module_name in source_code:
                        continue

                    module = sys.modules.get(module_name)
                    if module and hasattr(module, "__file__") and module.__file__:
                        with open(module.__file__, "r") as f:
                            source_code[module_name] = f.read()
                except Exception as e:
                    logger.warning(f"Could not capture source for component {name}: {e}")

        # Also capture key dependencies
        for module_name in [
            "metta.agent.lib.metta_module",
            "metta.agent.lib.metta_layer",
            "metta.agent.util.distribution_utils",
            "metta.agent.policy_state",
        ]:
            try:
                if module_name in source_code:
                    continue

                module = sys.modules.get(module_name)
                if module and hasattr(module, "__file__") and module.__file__:
                    with open(module.__file__, "r") as f:
                        source_code[module_name] = f.read()
            except Exception as e:
                logger.warning(f"Could not capture source for {module_name}: {e}")

        return source_code

    def build_from_pytorch_policy(self, path) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from a PyTorch policy checkpoint.

        Returns:
            Tuple of (MettaAgent, BuildContext) for reconstruction
        """
        from metta.agent.policy_store import build_pytorch_policy

        policy = build_pytorch_policy(path, self._cfg.device, pytorch=self._cfg.get("pytorch", None))

        # Create build context
        build_context = BuildContext(
            method="build_from_pytorch_policy",
            args=(path,),
            kwargs={"device": self._cfg.device, "pytorch": self._cfg.get("pytorch", None)},
        )

        return MettaAgent(policy), build_context

    def build_from_policy_class(self, policy_class, *args, **kwargs) -> Tuple["MettaAgent", BuildContext]:
        """Build a MettaAgent from a custom policy class.

        Returns:
            Tuple of (MettaAgent, BuildContext) for reconstruction
        """
        from metta.agent.pytorch_policy import PytorchPolicy

        policy = policy_class(*args, **kwargs)

        # Ensure it's a PytorchPolicy or wrap it if needed
        if not isinstance(policy, PytorchPolicy):
            logger.warning(f"Policy {policy_class.__name__} does not inherit from PytorchPolicy. Wrapping it.")
            policy = PytorchPolicy(policy)

        # Create build context with class info
        build_context = BuildContext(
            method="build_from_policy_class", args=args, kwargs={**kwargs, "class_name": policy_class.__name__}
        )

        return MettaAgent(policy), build_context

    def build_from_source_file(self, path: str) -> "MettaAgent":
        """Build a MettaAgent by reconstructing it from a saved source file.

        This is now deprecated in favor of using PolicyRecord.load() which
        handles reconstruction via BuildContext.
        """
        logger.warning("build_from_source_file is deprecated. Use PolicyRecord.load() instead.")

        # Import PolicyRecord here to avoid circular imports
        from metta.agent.policy_record import PolicyRecord

        # Create a temporary PolicyRecord to handle loading
        pr = PolicyRecord(
            policy_store=None,  # Not needed for loading
            name="temp",
            uri=f"file://{path}",
            metadata={},
        )

        # Set a minimal policy store config
        pr._policy_store = type("obj", (object,), {"_cfg": self._cfg})

        return pr.load(path, self._cfg.device)


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig) -> Tuple["MettaAgent", BuildContext]:
    """Create a policy with its build context.

    Returns:
        Tuple of (MettaAgent, BuildContext) for reconstruction
    """
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
