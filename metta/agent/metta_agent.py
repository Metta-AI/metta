import logging
from typing import TYPE_CHECKING, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.lib.obs_feature_remapper import ObsFeatureRemapper
from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.omegaconf import convert_to_dict

if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: "MettaGridEnv", cfg: ListConfig | DictConfig):
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Here's where we create MettaAgent. We're including the term MettaAgent here for better
    # searchability. Otherwise you might only find yaml files.
    return hydra.utils.instantiate(
        cfg.agent,
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False,
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

    def initialize_to_environment(
        self, features: dict[str, dict], action_names: list[str], action_max_params: list[int], device: torch.device
    ) -> None:
        return self.module.initialize_to_environment(features, action_names, action_max_params, device)


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
        component_cfgs = convert_to_dict(cfg.components)

        # First pass: instantiate all configured components
        for component_key in component_cfgs:
            # Convert key to string to ensure compatibility
            component_name = str(component_key)
            component_cfgs[component_key]["name"] = component_name
            logger.info(f"calling hydra instantiate from MettaAgent __init__ for {component_name}")
            component = hydra.utils.instantiate(component_cfgs[component_key], **self.agent_attributes)
            self.components[component_name] = component

        # Inject ObsFeatureRemapper after _obs_ if using token observations
        if "_obs_" in self.components:
            # Check if this is a token-based observation pipeline
            obs_component = self.components["_obs_"]
            if hasattr(obs_component, "__class__") and "ObsTokenPadStrip" in obs_component.__class__.__name__:
                logger.info("Injecting ObsFeatureRemapper for token-based observations")

                # Create the remapper component
                remapper = ObsFeatureRemapper(name="_obs_feature_remapper_")
                self.components["_obs_feature_remapper_"] = remapper

                # Rewire the observation pipeline
                # Find all components that use _obs_ as a source and update them to use the remapper
                for name, component in self.components.items():
                    if hasattr(component, "_sources") and component._sources is not None:
                        for source in component._sources:
                            if source.get("name") == "_obs_":
                                logger.info(f"Rewiring {name} to use _obs_feature_remapper_ instead of _obs_")
                                source["name"] = "_obs_feature_remapper_"

                # Set the remapper's source to _obs_
                remapper._sources = [{"name": "_obs_"}]

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

        # Initialize feature embedding maps
        self.feature_name_to_embedding: dict[str, torch.Tensor] = {}

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
        self, features: dict[str, dict], action_names: list[str], action_max_params: list[int], device
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
        """
        self._initialize_observations(features, device)
        self._initialize_actions(action_names, action_max_params, device)

    def _initialize_observations(self, features: dict[str, dict], device):
        """
        Initialize observation features by creating embeddings and mapping them to feature IDs.

        Args:
            features: Dictionary mapping feature names to their properties
            device: Device to place tensors on
        """
        self.active_features = features
        self.device = device

        # Create feature_id to feature_name mapping for quick lookup
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}

        # Store type and normalization information
        self.feature_types = {props["id"]: props["type"] for props in features.values()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if props["type"] == "scalar"
        }

        # CRITICAL: Store the original feature mapping on first initialization
        # This is used to remap feature IDs from new environments to match what the model was trained with
        if not hasattr(self, "original_feature_mapping"):
            # First time initialization - store the mapping
            self.original_feature_mapping = {}
            for name, props in features.items():
                self.original_feature_mapping[name] = props["id"]
            logger.info(f"Stored original feature mapping: {self.original_feature_mapping}")
        else:
            # Subsequent initialization - create remapping table
            self._create_feature_remapping_table(features)

        logger.info(f"Initialized observations with {len(features)} features")

    def _create_feature_remapping_table(self, features: dict[str, dict]):
        """
        Create a remapping table to translate new feature IDs to original ones.

        This is crucial for allowing trained models to work in environments where
        feature IDs might be different (e.g., "mineral" might be ID 5 instead of ID 3).
        """
        # Create a 256-element tensor that maps new IDs to original IDs
        # Initialize with identity mapping
        self.feature_id_remap = torch.arange(256, dtype=torch.uint8, device=self.device)

        # For each feature name, map the new ID to the original ID
        remapped_count = 0
        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
                    remapped_count += 1
                    logger.info(f"Remapping feature '{name}': new_id={new_id} -> original_id={original_id}")
            else:
                logger.warning(f"Feature '{name}' with id={new_id} not in original mapping, keeping as-is")

        if remapped_count > 0:
            logger.info(f"Created feature remapping table with {remapped_count} remapped features")

            # Update the observation processing components with the remapping
            if "_obs_feature_remapper_" in self.components:
                self.components["_obs_feature_remapper_"].update_remapping(self.feature_id_remap)

            # Update feature normalizations for downstream components
            # When feature IDs are remapped, we need to ensure normalization values
            # are correctly mapped to the original feature IDs
            original_normalizations = {}
            for name, props in features.items():
                if name in self.original_feature_mapping:
                    original_id = self.original_feature_mapping[name]
                    if "normalization" in props:
                        original_normalizations[original_id] = props["normalization"]

            # Update ObsAttrValNorm if it exists
            for comp_name, component in self.components.items():
                if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                    logger.info(f"Updating feature normalizations for {comp_name}")
                    # Create updated normalization tensor with remapped IDs
                    norm_tensor = torch.ones(256, dtype=torch.float32)
                    for feat_id, norm_val in original_normalizations.items():
                        norm_tensor[feat_id] = norm_val
                    component.register_buffer("_norm_factors", norm_tensor)

    def _initialize_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Initialize action configuration."""
        assert isinstance(action_max_params, list), "action_max_params must be a list"

        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names

        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")
        self.components["_action_embeds_"]._initialize_actions(full_action_names, self.device)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"Agent actions initialized with: {self.active_actions}")

    def get_feature_embeddings_for_checkpoint(self) -> dict[str, torch.Tensor]:
        """
        Get feature embeddings to include in checkpoint metadata.
        This reads from the current embedding tensors and maps them back to feature names.

        Returns:
            Dictionary mapping feature names to their current embedding tensors
        """
        # TODO: This will be implemented when we add actual feature embeddings
        # For now, return the stored embeddings if any
        return self.feature_name_to_embedding.copy()

    def restore_feature_embeddings_from_checkpoint(self, embeddings: dict[str, torch.Tensor]):
        """
        Restore feature embeddings from checkpoint metadata.

        Args:
            embeddings: Dictionary mapping feature names to embedding tensors
        """
        # TODO: This will be implemented when we add actual feature embeddings
        # For now, just store them
        self.feature_name_to_embedding = embeddings.copy()

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
