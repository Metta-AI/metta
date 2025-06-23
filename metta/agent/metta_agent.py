import logging
from typing import Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pufferlib.pytorch import sample_logits
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.util.omegaconf import convert_to_dict

logger = logging.getLogger("metta_agent")


class MettaAgent(nn.Module):
    """Unified neural network policy class that handles both component-based and PyTorch policies.

    This class can operate in two modes:
    1. Component mode: Complex policy built from modular components (former ComponentPolicy)
    2. PyTorch mode: Wrapper for external PyTorch policies (former PytorchPolicy)
    """

    def __init__(
        self,
        # For component mode
        obs_space: Optional[Union[gym.spaces.Space, gym.spaces.Dict]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        action_space: Optional[gym.spaces.Space] = None,
        feature_normalizations: Optional[list[float]] = None,
        device: Optional[str] = None,
        # For pytorch mode
        wrapped_policy: Optional[nn.Module] = None,
        **cfg,
    ):
        super().__init__()

        # Determine mode based on arguments
        if wrapped_policy is not None:
            # PyTorch policy mode - wrap an external policy
            self._init_pytorch_policy(wrapped_policy)
        else:
            # Component policy mode - build from components
            self._init_component_policy(
                obs_space, obs_width, obs_height, action_space, feature_normalizations, device, **cfg
            )

    def _init_pytorch_policy(self, wrapped_policy: nn.Module):
        """Initialize as a wrapper for an external PyTorch policy."""
        self.policy = wrapped_policy
        self._policy_hidden_size = getattr(wrapped_policy, "hidden_size", 256)
        self.clip_range = 0  # No clipping for PyTorch policies
        self.components = nn.ModuleDict()  # Empty for PyTorch policies

    def _init_component_policy(
        self, obs_space, obs_width, obs_height, action_space, feature_normalizations, device, **cfg
    ):
        """Initialize as a component-based policy."""
        # Note that this doesn't instantiate the components -- that will happen later once
        # we've built up the right parameters for them.
        cfg = OmegaConf.create(cfg)

        logger.info(f"obs_space: {obs_space} ")

        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range

        assert hasattr(cfg.observations, "obs_key") and cfg.observations.obs_key is not None, (
            "Configuration is missing required field 'observations.obs_key'"
        )
        obs_key = cfg.observations.obs_key  # typically "grid_obs"

        obs_shape = safe_get_from_obs_space(obs_space, obs_key, "shape")

        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": action_space,
            "feature_normalizations": feature_normalizations,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "obs_key": cfg.observations.obs_key,
            "obs_shape": obs_shape,
            "hidden_size": self.hidden_size,
            "core_num_layers": self.core_num_layers,
            "obs_space": obs_space,
            "device": device,
        }

        logging.info(f"agent_attributes: {self.agent_attributes}")

        self.components = nn.ModuleDict()
        component_cfgs = convert_to_dict(cfg.components)

        for component_key in component_cfgs:
            # Convert key to string to ensure compatibility
            component_name = str(component_key)
            component_cfgs[component_key]["name"] = component_name
            logger.info(f"calling hydra instantiate from MettaAgent __init__ for {component_name}")
            component = hydra.utils.instantiate(component_cfgs[component_key], **self.agent_attributes)
            self.components[component_name] = component

        component = self.components["_value_"]
        self._setup_components(component)
        component = self.components["_action_"]
        self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(
                    f"Component {name} in MettaAgent was never setup. It might not be accessible by other components."
                )

        self.components = self.components.to(device)

        self._total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")

    def _setup_components(self, component):
        """_sources is a list of dicts albeit many layers simply have one element.
        It must always have a "name" and that name should be the same as the relevant key in self.components.
        source_components is a dict of components that are sources for the current component. The keys
        are the names of the source components."""
        # recursively setup all source components
        if component._sources is not None:
            for source in component._sources:
                print(f"setting up source {source} with name {source['name']}")
                self._setup_components(self.components[source["name"]])

        # setup the current component and pass in the source components
        source_components = None
        if component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Run this at the beginning of training."""
        assert isinstance(action_max_params, list), "action_max_params must be a list"

        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names

        # For wrapped policies, delegate if they have this method
        wrapped_policy = getattr(self, "policy", None)
        if wrapped_policy and hasattr(wrapped_policy, "activate_actions"):
            wrapped_policy.activate_actions(action_names, action_max_params, device)

        # If no components, we're done
        if not hasattr(self, "components") or len(self.components) == 0:
            logger.info("MettaAgent actions activated")
            return

        # Component policy mode continues with full setup
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=self.device), dim=0)

        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        if "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, self.device)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"MettaAgent (Component mode) actions activated with: {self.active_actions}")

    @property
    def lstm(self):
        """Return the LSTM module if available."""
        # Check wrapped policy first
        if hasattr(self, "policy") and hasattr(self.policy, "lstm"):
            return self.policy.lstm
        # Then check components
        if hasattr(self, "components") and "_core_" in self.components:
            return self.components["_core_"]._net
        return None

    @property
    def is_pytorch_policy(self):
        """Check if this is a PyTorch policy (has a wrapped policy)."""
        return hasattr(self, "policy")

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def hidden_size(self):
        """Return the hidden size of the policy."""
        # For wrapped policies
        if hasattr(self, "_policy_hidden_size"):
            return self._policy_hidden_size
        # For component policies
        return getattr(self, "hidden_size", 256)

    @hidden_size.setter
    def hidden_size(self, value):
        """Set the hidden size (for backward compatibility with PyTorch policies)."""
        if hasattr(self, "policy"):
            self._policy_hidden_size = value

    @property
    def core_num_layers(self):
        """Return the number of LSTM layers."""
        # Check wrapped policy first
        if hasattr(self, "policy"):
            if hasattr(self.policy, "lstm") and hasattr(self.policy.lstm, "num_layers"):
                return self.policy.lstm.num_layers
            if hasattr(self.policy, "core_num_layers"):
                return self.policy.core_num_layers
        # For component policies
        return getattr(self, "core_num_layers", 2)

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass - delegates to appropriate method based on available attributes."""
        # If we have a wrapped policy, use PyTorch forward
        if hasattr(self, "policy"):
            return self._forward_pytorch(x, state, action)
        # Otherwise use component forward
        else:
            return self._forward_component(x, state, action)

    def _forward_pytorch(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for wrapped PyTorch policies."""
        # Get the raw output from the wrapped policy
        result = self.policy(x, state, action)

        # Check if it's a legacy policy that returns (hidden, critic)
        if len(result) == 2 and result[0].dim() >= 2:
            # Legacy policy interface: (hidden, critic) -> (action, logprob, entropy, value, logits)
            hidden, critic = result

            # Use pufferlib's sample_logits to handle action sampling
            action_result, logprob, logits_entropy = sample_logits(hidden, action)

            # Return in MettaAgent format
            # hidden -> log_probs (the raw logits)
            # critic -> value
            return action_result, logprob, logits_entropy, critic, hidden

        # Check if it's a policy that returns the full 5-tuple
        elif len(result) == 5:
            return result

        # Handle other policies that might return (logits, value)
        elif len(result) == 2:
            logits, value = result
            if action is None:
                # Inference mode
                action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)
                # Assume action is already in the correct format
                action = action_logit_index.unsqueeze(-1).repeat(1, 2)  # Dummy expansion
            else:
                # Training mode - need to evaluate given actions
                # This is a simplified version - real implementation would need proper action conversion
                action_log_prob, entropy, log_probs = evaluate_actions(logits, action[:, 0])

            return action, action_log_prob, entropy, value, log_probs

        else:
            raise ValueError(f"Wrapped policy returned unexpected number of values: {len(result)}")

    def _forward_component(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for component-based policies."""
        # Initialize dictionary for TensorDict
        td = {"x": x, "state": None}

        # Safely handle LSTM state
        if state.lstm_h is not None and state.lstm_c is not None:
            # Ensure states are on the same device as input
            lstm_h = state.lstm_h.to(x.device)
            lstm_c = state.lstm_c.to(x.device)
            # Concatenate LSTM states along dimension 0
            td["state"] = torch.cat([lstm_h, lstm_c], dim=0)

        # Forward pass through value network
        self.components["_value_"](td)
        value = td["_value_"]

        # Forward pass through action network
        self.components["_action_"](td)
        logits = td["_action_"]

        # Update LSTM states
        split_size = self.core_num_layers
        state.lstm_h = td["state"][:split_size]
        state.lstm_c = td["state"][split_size:]

        if action is None:
            return self._forward_inference(value, logits)
        else:
            return self._forward_training(value, logits, action)

    def _forward_inference(
        self, value: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference mode - samples new actions based on the policy."""
        # Sample actions
        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)

        # Convert logit index to action
        action = self._convert_logit_index_to_action(action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

    def _forward_training(
        self, value: torch.Tensor, logits: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training mode - evaluates the policy on provided actions."""
        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete action indices."""
        # Check wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "_convert_action_to_logit_index"):
            return policy._convert_action_to_logit_index(flattened_action)

        # Default implementation for policies without this method
        if not hasattr(self, "cum_action_max_params"):
            return flattened_action[:, 0]  # Just return action type

        # Component policy implementation
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        return action_logit_indices

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        # Check wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "_convert_logit_index_to_action"):
            return policy._convert_logit_index_to_action(action_logit_index)

        # Default implementation for policies without this method
        if not hasattr(self, "action_index_tensor"):
            return action_logit_index.unsqueeze(-1).repeat(1, 2)  # Dummy expansion

        # Component policy implementation
        action = self.action_index_tensor[action_logit_index]
        return action

    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """Apply a method to all components."""
        if not hasattr(self, "components") or len(self.components) == 0:
            return []

        results = []
        for name, component in self.components.items():
            if not hasattr(component, method_name):
                continue  # Skip components that don't have this method

            method = getattr(component, method_name)
            if not callable(method):
                raise TypeError(f"Component '{name}' has {method_name} attribute but it's not callable")

            result = method(*args, **kwargs)
            if result is not None:
                results.append(result)

        return results

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss."""
        # Try delegating to wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "l2_reg_loss"):
            return policy.l2_reg_loss()

        # Apply to components
        component_loss_tensors = self._apply_to_components("l2_reg_loss")
        if len(component_loss_tensors) > 0:
            return torch.sum(torch.stack(component_loss_tensors))

        # Default to zero
        return torch.tensor(0.0, device=getattr(self, "device", "cpu"))

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss."""
        # Try delegating to wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "l2_init_loss"):
            return policy.l2_init_loss()

        # Apply to components
        component_loss_tensors = self._apply_to_components("l2_init_loss")
        if len(component_loss_tensors) > 0:
            return torch.sum(torch.stack(component_loss_tensors))

        # Default to zero
        return torch.tensor(0.0, device=getattr(self, "device", "cpu"))

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copy."""
        # Try delegating to wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "update_l2_init_weight_copy"):
            policy.update_l2_init_weight_copy()
            return

        # Apply to components
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self):
        """Clip weights based on clip_range."""
        # Try delegating to wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "clip_weights"):
            policy.clip_weights()
            return

        # Only clip if clip_range is positive
        if getattr(self, "clip_range", 0) > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        # Try delegating to wrapped policy first
        policy = getattr(self, "policy", None)
        if policy and hasattr(policy, "compute_weight_metrics"):
            return policy.compute_weight_metrics(delta)

        # Apply to components
        if not hasattr(self, "components"):
            return []

        results = {}
        for name, component in self.components.items():
            if not hasattr(component, "compute_weight_metrics"):
                continue  # Skip components that don't have this method

            method = component.compute_weight_metrics
            if callable(method):
                results[name] = method(delta)

        return [metrics for metrics in results.values() if metrics is not None]


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


# Factory methods for backward compatibility
def make_policy(env, cfg: Union[DictConfig, ListConfig]) -> MettaAgent:
    """Create a MettaAgent policy from environment and configuration.

    This is a convenience function for backward compatibility.

    Args:
        env: The environment with observation/action space information
        cfg: Configuration dict containing agent parameters

    Returns:
        MettaAgent instance in component mode
    """
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Create MettaAgent in component mode
    return hydra.utils.instantiate(
        cfg.agent,
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        device=cfg.device,
        _target_="metta.agent.metta_agent.MettaAgent",
        _recursive_=False,
    )


def make_distributed_agent(agent: Union[MettaAgent, nn.Module], device: torch.device) -> DistributedMettaAgent:
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
