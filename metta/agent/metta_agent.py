import logging
from typing import Any, List, Optional, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import sample_logits
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig):
    # Debug: Print env content
    print("=== PufferEnv DEBUG INFO ===")
    print(f"Type: {type(env)}")
    print(f"Dir: {dir(env)}")
    print(f"Single observation space: {env.single_observation_space}")
    print(f"Single action space: {env.single_action_space}")
    print(f"Grid features: {env.grid_features}")
    print(f"Global features: {env.global_features}")
    print("==========================")

    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )
    return hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False,
    )


class DistributedMettaAgent(DistributedDataParallel):
    def __init__(self, agent, device):
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def safe_get_obs_property(
    obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
    obs_key: str,
    property_index: Optional[int] = None,
    property_name: str = "shape",
) -> Any:
    """
    Safely extract properties from observation spaces with comprehensive error handling.

    Args:
        obs_space: The observation space to extract properties from
        obs_key: The key to access in the observation space
        property_index: Optional index to access in the property (e.g., 1 for shape[1:], 2 for shape[2])
        property_name: The name of the property to extract (default: "shape")

    Returns:
        The extracted property value

    Raises:
        ValueError: If the property cannot be safely extracted
    """
    try:
        if isinstance(obs_space, gym.spaces.Dict):
            if obs_key in obs_space.spaces:
                space = obs_space.spaces[obs_key]
                if space is None:
                    raise ValueError(f"Space for key '{obs_key}' is None")
                if not hasattr(space, property_name):
                    raise ValueError(f"Space for key '{obs_key}' has no {property_name} attribute")

                prop = getattr(space, property_name)
                if prop is None:
                    raise ValueError(f"{property_name.capitalize()} for space '{obs_key}' is None")

                if property_index is not None:
                    if len(prop) <= property_index:
                        raise ValueError(
                            f"{property_name.capitalize()} for space '{obs_key}' does not have an index {property_index}"
                        )
                    if property_index == 1:
                        return prop[1:]  # Special case for obs_input_shape
                    return prop[property_index]
                return prop
            else:
                raise ValueError(
                    f"Key '{obs_key}' not found in observation space. Available keys: {list(obs_space.spaces.keys())}"
                )
        elif hasattr(obs_space, property_name):
            prop = getattr(obs_space, property_name)
            if prop is None:
                raise ValueError(f"Observation space {property_name} is None")

            if property_index is not None:
                if len(prop) <= property_index:
                    raise ValueError(f"Observation space {property_name} does not have an index {property_index}")
                if property_index == 1:
                    return prop[1:]  # Special case for obs_input_shape
                return prop[property_index]
            return prop
        else:
            raise ValueError(f"Observation space doesn't have a {property_name} attribute")
    except (TypeError, AttributeError, IndexError) as e:
        raise ValueError(f"Error accessing {property_name} from observation space: {e}") from e


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        action_space: gym.spaces.Space,
        grid_features: List[str],
        device: str,
        **cfg,
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range

        try:
            obs_key = cfg.observations.obs_key
        except (AttributeError, KeyError) as err:
            raise ValueError("Configuration is missing required field 'observations.obs_key'") from err

        obs_input_shape = safe_get_obs_property(obs_space, obs_key, property_index=1)

        # this is hardcoded to index 2 to provide the channel # at end of tuple
        num_objects = safe_get_obs_property(obs_space, obs_key, property_index=2)

        agent_attributes = {
            "obs_shape": obs_shape,
            "clip_range": self.clip_range,
            "action_space": action_space,
            "grid_features": grid_features,
            "obs_key": cfg.observations.obs_key,
            "obs_input_shape": obs_input_shape,
            "num_objects": num_objects,
            "hidden_size": self.hidden_size,
            "core_num_layers": self.core_num_layers,
        }

        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)

        # Check if component_cfgs is a list or dictionary
        if isinstance(component_cfgs, list):
            # Handle list case
            for i, component_cfg in enumerate(component_cfgs):
                if isinstance(component_cfg, dict):
                    component_name = str(component_cfg.get("name", f"component_{i}"))
                    component_cfg["name"] = component_name
                    component = hydra.utils.instantiate(component_cfg, **agent_attributes)
                    self.components[component_name] = component
                else:
                    raise ValueError(f"Component configuration at index {i} is not a dictionary")
        elif isinstance(component_cfgs, dict):
            # Handle dictionary case (original code)
            for component_key in component_cfgs:
                # Convert key to string to ensure compatibility
                component_name = str(component_key)
                component_cfgs[component_key]["name"] = component_name
                component = hydra.utils.instantiate(component_cfgs[component_key], **agent_attributes)
                self.components[component_name] = component
        else:
            raise ValueError(f"Component configurations must be a list or dictionary, got {type(component_cfgs)}")

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
                self._setup_components(self.components[source["name"]])

        # setup the current component and pass in the source components
        source_components = None
        if component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def activate_actions(self, action_names, action_max_params, device):
        """Run this at the beginning of training."""
        self.device = device
        self.actions_max_params = action_max_params
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=self.device), dim=0)

        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")
        self.components["_action_embeds_"].activate_actions(full_action_names, self.device)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device)
        logger.info(f"Agent actions activated with: {self.active_actions}")

    @property
    def lstm(self):
        return self.components["_core_"]._net

    @property
    def total_params(self):
        return self._total_params

    def forward(self, x, state: PolicyState, action=None):
        """
        Forward pass of the MettaAgent.

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states
            action: Optional action tensor

        Returns:
            Tuple of (action, logprob_act, entropy, value, log_sftmx_logits)
        """
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
        if td["state"] is not None:
            split_size = self.core_num_layers
            if split_size <= td["state"].shape[0]:
                state.lstm_h = td["state"][:split_size]
                state.lstm_c = td["state"][split_size:]
            else:
                # Handle error case where state tensor is smaller than expected
                raise ValueError(
                    "State tensor has insufficient size: "
                    f"{td['state'].shape[0]} < {split_size * 2} (expected for h and c)"
                )

        # Sample actions
        action_logit_index = self._convert_action_to_logit_index(action) if action is not None else None
        action_logit_index, logprob_act, entropy, log_sftmx_logits = sample_logits(logits, action_logit_index)

        # Convert logit index to action if no action was provided
        if action is None:
            action = self._convert_logit_index_to_action(action_logit_index, td)

        return action, logprob_act, entropy, value, log_sftmx_logits

    def _convert_action_to_logit_index(self, action):
        """Convert action pairs (action_type, action_param) to single discrete action logit indices using vectorized
        operations"""
        action = action.reshape(-1, 2)

        # Extract action components
        action_type_numbers = action[:, 0].long()
        action_params = action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]

        # Vectorized addition
        action_logit_index = action_type_numbers + cumulative_sum + action_params

        return action_logit_index.reshape(-1, 1)

    def _convert_logit_index_to_action(self, action_logit_index, td):
        """Convert logit indices back to action pairs using tensor indexing"""
        # direct tensor indexing on precomputed action_index_tensor
        return self.action_index_tensor[action_logit_index.reshape(-1)]

    def _apply_to_components(self, method_name, *args, **kwargs):
        """
        Safely apply a method to all components.

        Args:
            method_name: Name of the method to call on each component
            *args, **kwargs: Arguments to pass to the method

        Returns:
            Dictionary of component names mapped to their return values
        """
        results = {}
        for name, component in self.components.items():
            try:
                if hasattr(component, method_name):
                    method = getattr(component, method_name)
                    if callable(method):
                        results[name] = method(*args, **kwargs)
                    else:
                        logger.warning(f"Component '{name}' has {method_name} attribute but it's not callable")
                # Otherwise skip this component
            except Exception as e:
                logger.warning(f"Error applying {method_name} to component '{name}': {e}")

        return results

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust
        it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        # Initialize with a tensor of zeros
        loss_value = torch.tensor(0.0, device=self.device if hasattr(self, "device") else None)

        # Use the helper method to gather all component losses
        component_losses = self._apply_to_components("l2_reg_loss")

        # Process the results
        for name, comp_loss in component_losses.items():
            if comp_loss is not None:
                # Convert to tensor if it's not already
                if not isinstance(comp_loss, torch.Tensor):
                    comp_loss = torch.tensor(float(comp_loss), device=loss_value.device)
                loss_value = loss_value + comp_loss

        return loss_value

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        # Initialize with a tensor of zeros
        loss_value = torch.tensor(0.0, device=self.device if hasattr(self, "device") else None)

        # Use the helper method to gather all component losses
        component_losses = self._apply_to_components("l2_init_loss")

        # Process the results
        for name, comp_loss in component_losses.items():
            if comp_loss is not None:
                # Convert to tensor if it's not already
                if not isinstance(comp_loss, torch.Tensor):
                    comp_loss = torch.tensor(float(comp_loss), device=loss_value.device)
                loss_value = loss_value + comp_loss

        return loss_value

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval. 0 means no updating."""
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self):
        """Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value
        effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global
        loss value or 0 to turn it off."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        """Compute weight metrics for all components that have weights enabled for analysis.
        Returns a list of metric dictionaries, one per component. Set analyze_weights to True in the config to turn it
        on for a given component."""
        results = self._apply_to_components("compute_weight_metrics", delta)
        return [metrics for metrics in results.values() if metrics is not None]
