import logging
from typing import List, Optional, Union

import einops
import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import sample_logits
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.util.omegaconf import convert_to_dict
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


# Helper class for PufferLib Experience buffer compatibility
class _RecurrentStateProxy:
    def __init__(self, hidden_size: int, num_layers: int = 1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig):
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    return hydra.utils.instantiate(
        cfg.agent,
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


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        action_space: gym.spaces.Space,
        grid_features: List[str],
        device: str,
        **cfg,
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        print("<<<< DEBUG: MettaAgent __init__ v_TRANSFORMER_DEBUG_001 >>>>")
        self._transformer_debug_flag = "v_TRANSFORMER_DEBUG_001"

        logger.info(f"obs_space: {obs_space} ")

        core_cfg = cfg.components._core_
        self.hidden_size = core_cfg.output_size  # This is d_model for Transformer
        # num_mem_tokens is needed for _RecurrentStateProxy
        # It should be part of the _core_ component's config (e.g., core_cfg.num_mem_tokens)
        # Ensure your YAML for TransformerMemoryCore specifies num_mem_tokens.
        num_mem_tokens = core_cfg.get("num_mem_tokens", 1)  # Default to 1 if not in config, though it should be
        if not hasattr(core_cfg, "num_mem_tokens"):
            logger.warning(
                "num_mem_tokens not found in core_cfg, defaulting to 1 for _RecurrentStateProxy. Please specify in YAML."
            )

        self.clip_range = cfg.clip_range

        # Add this attribute for PufferLib Experience buffer compatibility
        # num_layers for the proxy will be num_mem_tokens for the Transformer state
        self.recurrent_state_proxy = _RecurrentStateProxy(hidden_size=self.hidden_size, num_layers=num_mem_tokens)

        assert hasattr(cfg.observations, "obs_key") and cfg.observations.obs_key is not None, (
            "Configuration is missing required field 'observations.obs_key'"
        )
        obs_key = cfg.observations.obs_key  # typically "grid_obs"

        obs_shape = safe_get_from_obs_space(obs_space, obs_key, "shape")  # obs_w, obs_h, num_objects
        num_objects = obs_shape[2]

        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": action_space,
            "grid_features": grid_features,
            "obs_key": cfg.observations.obs_key,
            "obs_shape": obs_shape,
            "num_objects": num_objects,
            "hidden_size": self.hidden_size,
            # "core_num_layers": self.core_num_layers,
        }

        logging.info(f"agent_attributes: {self.agent_attributes}")

        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

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

    def get_initial_policy_state(self, batch_size: Optional[int] = None) -> PolicyState:
        """Returns an initial PolicyState for the agent for a given batch_size."""
        # For TransformerMemoryCore, initial memory_tokens are best handled internally if None.
        # memory_tokens=None is the key, hidden field removed from PolicyState.
        return PolicyState(memory_tokens=None)

    @property
    def total_params(self):
        return self._total_params

    def forward(self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None):
        """
        Forward pass of the MettaAgent.

        1. Inference mode (action=None): sample new actions based on the policy
        - x shape: (BT, *self.obs_shape)
        - Output action shape: (BT, 1) -- we return the action index rather than the (type, arg) tuple

        2. BPTT training mode (action is provided): evaluate the policy on past actions
        - x shape: (B, T, *self.obs_shape)
        - action shape: (B, T, 2)
        - Output action shape: (B, T, 2) -- we return the (type, arg) tuple

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states
            action: Optional action tensor for BPTT

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Sampled output action (inference) or same as input action (BPTT)
            - action_log_prob: Log probability of the output action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT,)
            - log_probs: Log-softmax of logits, shape (BT, A) where A is the size of the action space
        """
        # rename parameter for clarity
        bptt_action = action
        del action

        # TODO - obs_shape is not available in the eval smoke test policies so we can't
        # check exact dimensions

        # Default values in case obs_shape is not available
        obs_w, obs_h, features = "W", "H", "F"

        # Check if agent_attributes exists, is not None, and contains obs_shape
        if (
            hasattr(self, "agent_attributes")
            and self.agent_attributes is not None
            and "obs_shape" in self.agent_attributes
        ):
            # Get obs_shape and ensure it has the expected format
            obs_shape = self.agent_attributes["obs_shape"]
            if isinstance(obs_shape, (list, tuple)) and len(obs_shape) == 3:
                obs_w, obs_h, features = obs_shape
            # If the format is unexpected, we keep the default values

        if bptt_action is not None:
            # BPTT
            if __debug__:
                assert_shape(bptt_action, ("B", "T", 2), "bptt_action")

            B, T, A = bptt_action.shape

            if __debug__:
                assert A == 2, f"Action dimensionality should be 2, got {A}"
                assert_shape(x, (B, T, obs_w, obs_h, features), "x")

            # Flatten batch and time dimensions for both action and x
            bptt_action = einops.rearrange(bptt_action, "b t c -> (b t) c")
            x = einops.rearrange(x, "b t ... -> (b t) ...")

            if __debug__:
                assert_shape(bptt_action, (B * T, 2), "flattened action")
                assert_shape(x, (B * T, obs_w, obs_h, features), "flattened x")
        else:
            # inference
            if __debug__:
                assert_shape(x, ("BT", obs_w, obs_h, features), "x")

        # Initialize dictionary for TensorDict
        td = {"x": x, "state": None}

        # Safely handle recurrent state (now memory_tokens)
        if state.memory_tokens is not None:
            # Ensure memory_tokens are on the same device as input
            # The TransformerMemoryCore will also handle moving its state to device,
            # but good practice here too.
            td["state"] = state.memory_tokens.to(x.device)
        # else: td["state"] remains None, TransformerMemoryCore will initialize its state.

        # Forward pass through value network. Relies on _core_ populating td["core_value_output"].
        self.components["_value_"](td)
        value = td["_value_"]

        # Forward pass through action network. Relies on _core_ populating td["core_action_output"].
        self.components["_action_"](td)
        logits = td["_action_"]

        # Update recurrent state from td, which was updated by TransformerMemoryCore
        if "state" in td and td["state"] is not None:
            state.memory_tokens = td["state"]  # td["state"] is already detached by TransformerMemoryCore
        else:
            state.memory_tokens = None

        # Sample actions
        if bptt_action is not None:
            # BPTT
            bptt_action_index = self._convert_action_to_logit_index(bptt_action)

            if __debug__:
                action_space_size = logits.shape[-1]  # 'A' dimension size
                max_index = bptt_action_index.max().item()
                min_index = bptt_action_index.min().item()
                if max_index >= action_space_size or min_index < 0:
                    raise ValueError(
                        f"Invalid action_logit_index: contains values outside the valid range"
                        f" [0, {action_space_size - 1}]. "
                        f"Found values in range [{min_index}, {max_index}]"
                    )

            action_index, action_log_prob, entropy, log_probs = sample_logits(logits, bptt_action_index)
        else:
            # inference
            action_index, action_log_prob, entropy, log_probs = sample_logits(logits, None)

        if __debug__:
            assert_shape(action_index, ("BT",), "action_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")
            assert_shape(entropy, ("BT",), "entropy")
            assert_shape(log_probs, ("BT", "A"), "log_probs")

        output_action = self._convert_logit_index_to_action(action_index)
        if __debug__:
            assert_shape(output_action, ("BT", 2), "output_action")

        return output_action, action_log_prob, entropy, value, log_probs

    def _convert_action_to_logit_index(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete action indices
        using precomputed offsets.

        Args:
            action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs

        Returns:
            action_logit_indices: Tensor of shape [B*T] containing flattened action indices
        """
        if __debug__:
            assert_shape(action, ("BT", 2), "action")

        action_type_numbers = action[:, 0].long()
        action_params = action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        if __debug__:
            assert_shape(action_logit_indices, ("BT",), "action_logit_indices")

        return action_logit_indices  # shape: [B*T]

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs using tensor indexing.

        Args:
            action_logit_index: Tensor of shape [B*T] containing flattened action indices

        Returns:
            action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs
        """
        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")

        action = self.action_index_tensor[action_logit_index]

        if __debug__:
            assert_shape(action, ("BT", 2), "actions")

        return action

    def _apply_to_components(self, method_name, *args, **kwargs) -> List[torch.Tensor]:
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
        assert len(self.components) != 0, "No components available to apply method"

        results = []
        for name, component in self.components.items():
            if not hasattr(component, method_name):
                raise AttributeError(f"Component '{name}' does not have method '{method_name}'")

            method = getattr(component, method_name)
            if not callable(method):
                raise TypeError(f"Component '{name}' has {method_name} attribute but it's not callable")

            results.append(method(*args, **kwargs))

        return results

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust
        it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        component_loss_tensors = self._apply_to_components("l2_reg_loss")
        return torch.sum(torch.stack(component_loss_tensors))

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        component_loss_tensors = self._apply_to_components("l2_init_loss")
        return torch.sum(torch.stack(component_loss_tensors))

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
        results = {}
        for name, component in self.components.items():
            method_name = "compute_weight_metrics"
            if not hasattr(component, method_name):
                continue  # Skip components that don't have this method instead of raising an error

            method = getattr(component, method_name)
            assert callable(method), f"Component '{name}' has {method_name} attribute but it's not callable"

            results[name] = method(delta)

        metrics_list = [metrics for metrics in results.values() if metrics is not None]
        return metrics_list
