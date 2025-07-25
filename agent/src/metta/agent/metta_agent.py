import logging
from typing import TYPE_CHECKING, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.registry import ActionRegistry, FeatureRegistry
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.component_factory import instantiate

if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


def make_policy(env: "MettaGridEnv", cfg: DictConfig) -> "MettaAgent":
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Convert agent config to dict for unpacking as kwargs
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)

    # Create MettaAgent directly without Hydra
    return MettaAgent(
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        global_features=env.global_features,
        device=cfg.device,  # cfg.device is required
        **agent_cfg,
    )


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

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Initialize the policy to the current environment's features and actions.
        This should be called exactly once per time the policy is "brought out of storage".

        Args:
            features: Dictionary mapping feature names to their properties:
                {
                    feature_name: {
                        "id": byte,  # The feature_id to use during this run
                        "type": "scalar" | "categorical",
                        "normalization": float (optional, only for scalar features)
                    }
                }
            action_names: List of action names
            action_max_params: List of maximum parameters for each action
            device: Device to place tensors on
            is_training: Deprecated. Training mode is now automatically detected.
        """
        self.device = device

        # Initialize features using registry
        remap_tensor, norm_tensor = self.feature_registry.initialize_with_features(features, device)

        # Apply feature remapping to observation component if it exists
        if "_obs_" in self.components and hasattr(self.components["_obs_"], "update_feature_remapping"):
            self.components["_obs_"].update_feature_remapping(remap_tensor)

        # Update normalization factors for components
        for _comp_name, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                component.register_buffer("_norm_factors", norm_tensor)

        # Initialize actions using registry
        self.action_names = action_names
        self.action_max_params = action_max_params

        # Get active action indices
        active_indices = self.action_registry.initialize_with_actions(action_names, action_max_params, device)

        # Update action embeddings component
        if "_action_embeds_" in self.components:
            action_embeds = self.components["_action_embeds_"]
            action_embeds.active_indices = active_indices
            action_embeds.num_actions = len(active_indices)

            # Ensure embedding layer is large enough
            action_embeds._net = self.action_registry.ensure_embedding_size(action_embeds._net)

        # Precompute cumulative sums for action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        # Create action index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])
        self.action_index_tensor = torch.tensor(action_index, device=device, dtype=torch.int32)


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        obs_width: int,
        obs_height: int,
        action_space: gym.spaces.Space,
        feature_normalizations: dict[int, float],
        device: str,
        **cfg,
    ):
        super().__init__()
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
        }

        logging.info(f"agent_attributes: {self.agent_attributes}")

        self.components = nn.ModuleDict()
        # Keep component configs as DictConfig to support both dict and attribute access
        component_cfgs = cfg.components

        # First pass: instantiate all configured components
        for component_key in component_cfgs:
            # Convert key to string to ensure compatibility
            component_name = str(component_key)

            # Convert to dict and merge attributes for instantiation
            comp_dict = dict(component_cfgs[component_key], **self.agent_attributes, name=component_name)

            # Instantiate component
            self.components[component_name] = instantiate(comp_dict)

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

        # Initialize registries for features and actions
        self.feature_registry = FeatureRegistry()
        self.action_registry = ActionRegistry()

    def _setup_components(self, component):
        """_sources is a list of dicts albeit many layers simply have one element.
        It must always have a "name" and that name should be the same as the relevant key in self.components.
        source_components is a dict of components that are sources for the current component. The keys
        are the names of the source components."""
        # recursively setup all source components
        if component._sources is not None:
            for source in component._sources:
                logger.info(f"setting up {component._name} with source {source['name']}")
                self._setup_components(self.components[source["name"]])

        # setup the current component and pass in the source components
        source_components = None
        if component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Initialize the policy to the current environment's features and actions.
        This should be called exactly once per time the policy is "brought out of storage".

        Args:
            features: Dictionary mapping feature names to their properties:
                {
                    feature_name: {
                        "id": byte,  # The feature_id to use during this run
                        "type": "scalar" | "categorical",
                        "normalization": float (optional, only for scalar features)
                    }
                }
            action_names: List of action names
            action_max_params: List of maximum parameters for each action
            device: Device to place tensors on
            is_training: Deprecated. Training mode is now automatically detected.
        """
        self.device = device

        # Initialize features using registry
        remap_tensor, norm_tensor = self.feature_registry.initialize_with_features(features, device)

        # Apply feature remapping to observation component if it exists
        if "_obs_" in self.components and hasattr(self.components["_obs_"], "update_feature_remapping"):
            self.components["_obs_"].update_feature_remapping(remap_tensor)

        # Update normalization factors for components
        for _comp_name, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                component.register_buffer("_norm_factors", norm_tensor)

        # Initialize actions using registry
        self.action_names = action_names
        self.action_max_params = action_max_params

        # Get active action indices
        active_indices = self.action_registry.initialize_with_actions(action_names, action_max_params, device)

        # Update action embeddings component
        if "_action_embeds_" in self.components:
            action_embeds = self.components["_action_embeds_"]
            action_embeds.active_indices = active_indices
            action_embeds.num_actions = len(active_indices)

            # Ensure embedding layer is large enough
            action_embeds._net = self.action_registry.ensure_embedding_size(action_embeds._net)

        # Precompute cumulative sums for action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        # Create action index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])
        self.action_index_tensor = torch.tensor(action_index, device=device, dtype=torch.int32)

    @property
    def lstm(self):
        return self.components["_core_"]._net

    @property
    def total_params(self):
        return self._total_params

    def forward_inference(
        self, value: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference mode - samples new actions based on the policy.

        Args:
            value: Value estimate tensor, shape (BT, 1)
            logits: Action logits tensor, shape (BT, A)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Sampled action, shape (BT, 2)
            - action_log_prob: Log probability of the sampled action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT, 1)
            - log_probs: Log-softmax of logits, shape (BT, A)
        """
        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        # Sample actions
        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")
            assert_shape(entropy, ("BT",), "entropy")
            assert_shape(log_probs, ("BT", "A"), "log_probs")

        # Convert logit index to action
        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_output_action")

        return action, action_log_prob, entropy, value, log_probs

    def forward_training(
        self, value: torch.Tensor, logits: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training mode - evaluates the policy on provided actions.

        Args:
            value: Value estimate tensor, shape (BT, 1)
            logits: Action logits tensor, shape (BT, A)
            action: Action tensor for evaluation, shape (B, T, 2)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Same as input action, shape (B, T, 2)
            - action_log_prob: Log probability of the provided action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT, 1)
            - log_probs: Log-softmax of logits, shape (BT, A)
        """
        if __debug__:
            assert_shape(value, ("BT", 1), "training_value")
            assert_shape(logits, ("BT", "A"), "training_logits")
            assert_shape(action, ("B", "T", 2), "training_input_action")

        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "converted_action_logit_index")

        action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        if __debug__:
            assert_shape(action_log_prob, ("BT",), "training_action_log_prob")
            assert_shape(entropy, ("BT",), "training_entropy")
            assert_shape(log_probs, ("BT", "A"), "training_log_probs")
            assert_shape(action, ("B", "T", 2), "training_output_action")

        return action, action_log_prob, entropy, value, log_probs

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MettaAgent - delegates to appropriate specialized method.

        Args:
            x: Input observation tensor
            state: Policy state containing LSTM hidden and cell states
            action: Optional action tensor for BPTT

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
        """
        if __debug__:
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

            # TODO: redo this and the above once we converge on token obs space. Commenting out for now.
            if action is None:
                # Inference: x should have shape (BT, obs_w, obs_h, features)
                pass
            else:
                # Training: x should have shape (B, T, obs_w, obs_h, features)
                B, T, A = action.shape
                assert A == 2, f"Action dimensionality should be 2, got {A}"
                # assert_shape(action, (B, T, 2), "training_input_action")

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

        # Value shape is (BT, 1) - keeping the final dimension explicit (instead of squeezing)
        # This design supports potential future extensions like distributional value functions
        # or multi-head value networks which would require more than a scalar per state
        if __debug__:
            assert_shape(value, ("BT", 1), "value")

        # Forward pass through action network
        self.components["_action_"](td)
        logits = td["_action_"]

        if __debug__:
            # here A is the size of the flattened action space (i.e. all valid (type, arg) combinations)
            assert_shape(logits, ("BT", "A"), "logits")

        # NOTE: Both value and logits always have shape (BT, *) regardless of input mode:
        # - Training input: (B, T, *obs_shape) gets internally reshaped to (BT, *) by LSTM
        # - Inference input: (BT, *obs_shape) stays as (BT, *)

        # Update LSTM states
        split_size = self.core_num_layers
        state.lstm_h = td["state"][:split_size]
        state.lstm_c = td["state"][split_size:]

        if action is None:
            return self.forward_inference(value, logits)
        else:
            return self.forward_training(value, logits, action)

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete action indices
        using precomputed offsets.

        Args:
            flattened_action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs

        Returns:
            action_logit_indices: Tensor of shape [B*T] containing flattened action indices
        """
        if __debug__:
            assert_shape(flattened_action, ("BT", 2), "flattened_action")

        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        if __debug__:
            assert_shape(action_logit_indices, ("BT",), "action_logit_indices")

        return action_logit_indices

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
        assert len(self.components) != 0, "No components available to apply method"

        results = []
        for name, component in self.components.items():
            if not hasattr(component, method_name):
                raise AttributeError(f"Component '{name}' does not have method '{method_name}'")

            method = getattr(component, method_name)
            if not callable(method):
                raise TypeError(f"Component '{name}' has {method_name} attribute but it's not callable")

            result = method(*args, **kwargs)
            if result is not None:
                results.append(result)

        return results

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        component_loss_tensors = self._apply_to_components("l2_init_loss")
        if len(component_loss_tensors) > 0:
            return torch.sum(torch.stack(component_loss_tensors))
        else:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval. 0 means no updating."""
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self):
        """Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value
        effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global
        loss value or 0 to turn it off."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
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
