import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import gymnasium as gym
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.datastruct import duplicates

logger = logging.getLogger(__name__)


class ComponentPolicy(nn.Module, ABC):
    """
    Abstract base class for component-based policies.
    Subclasses must override _build_components() to define their architecture.
    """

    def __init__(
        self,
        obs_space: Optional[Union[gym.spaces.Space, gym.spaces.Dict]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        feature_normalizations: Optional[dict[int, float]] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()

        # Store config parameters
        self.config = config or {}
        self.clip_range = self.config.get("clip_range", 0)
        self.analyze_weights_interval = self.config.get("analyze_weights_interval", 300)

        # Extract observation shape (always uses "grid_obs" key)
        obs_key = "grid_obs"
        obs_shape = safe_get_from_obs_space(obs_space, obs_key, "shape") if obs_space is not None else None

        # Create agent attributes dict
        self.agent_attributes = {
            "clip_range": self.clip_range,
            "feature_normalizations": feature_normalizations,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "obs_key": obs_key,
            "obs_shape": obs_shape,
        }

        self.components = nn.ModuleDict()

        # Build components using the abstract method
        components = self._build_components()

        # Add all components to the ModuleDict
        self.components.update(components)

        # Setup components by triggering leaf node setup
        for component_name in self._get_output_heads():
            component = self.components[component_name]
            self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(f"Component {name} in {self.__class__.__name__} policy was never setup.")

        # Check for duplicate component names
        all_names = [c._name for c in self.components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"{self.__class__.__name__} policy components: {self.components}")

        # Initialize action conversion tensors (will be set by MettaAgent)
        self.cum_action_max_params = None
        self.action_index_tensor = None

        logger.info(f"{self.__class__.__name__} policy initialized with components: {list(self.components.keys())}")

    @abstractmethod
    def _build_components(self) -> dict:
        """Build the component dictionary. Must be implemented by subclasses to define architecture."""
        pass

    @abstractmethod
    def _get_output_heads(self) -> list[str]:
        """Get the output heads for the policy."""
        pass

    def _setup_components(self, component):
        """Setup component connections."""
        # Skip if already setup
        if getattr(component, "ready", False):
            return

        # Recursively setup all source components first
        if hasattr(component, "_sources") and component._sources is not None:
            for source in component._sources:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logger.info(f"setting up {component._name} with source {source['name']}")
                self._setup_components(self.components[source["name"]])

        # Setup the current component and pass in the source components
        source_components = None
        if hasattr(component, "_sources") and component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass."""
        # Handle BPTT reshaping
        if td.batch_dims > 1:
            B = td.batch_size[0]
            TT = td.batch_size[1]
            td = td.reshape(td.batch_size.numel())  # flatten to BT
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size.numel()
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))

        # Run the value and action components
        self.components["_value_"](td)
        self.components["_action_"](td)

        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            # Reshape back for training mode
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            output_td = output_td.reshape(batch_size, bptt_size)

        return output_td

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """Sample actions for inference."""
        value = td["_value_"]
        logits = td["_action_"]

        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        action_logit_index, action_log_prob, _, full_log_probs = sample_actions(logits)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")

        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_action")

        td["actions"] = action
        td["act_log_prob"] = action_log_prob
        td["values"] = value.flatten()
        td["full_log_probs"] = full_log_probs
        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """Evaluate actions for training."""
        value = td["_value_"]
        logits = td["_action_"]

        if __debug__:
            assert_shape(value, ("BT", 1), "training_value")
            assert_shape(logits, ("BT", "A"), "training_logits")
            assert_shape(action, ("B", "T", 2), "training_input_action")

        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "converted_action_logit_index")

        action_log_prob, entropy, full_log_probs = evaluate_actions(logits, action_logit_index)

        if __debug__:
            assert_shape(action_log_prob, ("BT",), "training_action_log_prob")
            assert_shape(entropy, ("BT",), "training_entropy")
            assert_shape(full_log_probs, ("BT", "A"), "training_log_probs")

        td["act_log_prob"] = action_log_prob
        td["entropy"] = entropy
        td["value"] = value
        td["full_log_probs"] = full_log_probs
        return td

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        if flattened_action.size(0) == 0:
            raise ValueError("'flattened_action' dimension 0 ('BT') has invalid size 0, expected a positive value")
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def activate_action_embeddings(self, full_action_names: list[str], device):
        """Activate action embeddings with the given action names."""
        if "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, device)

    # ============================================================================
    # Memory-related Methods
    # ============================================================================

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def on_new_training_run(self):
        for _, component in self.components.items():
            component.on_new_training_run()

    def on_rollout_start(self):
        for _, component in self.components.items():
            component.on_rollout_start()

    def on_train_mb_start(self):
        for _, component in self.components.items():
            component.on_train_mb_start()

    def on_eval_start(self):
        for _, component in self.components.items():
            component.on_eval_start()

    # ============================================================================
    # Weight/Training Utility Methods
    # ============================================================================

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for all components."""
        results = {}
        for name, component in self.components.items():
            if hasattr(component, "compute_weight_metrics"):
                result = component.compute_weight_metrics(delta)
                if result is not None:
                    results[name] = result
        return list(results.values())

    # ============================================================================
    # Feature/Normalization Methods
    # ============================================================================

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping to observation component."""
        if "_obs_" in self.components:
            obs_component = self.components["_obs_"]
            if hasattr(obs_component, "update_feature_remapping"):
                obs_component.update_feature_remapping(remap_tensor)

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors for ObsAttrValNorm components after feature remapping."""
        for _, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                # Create normalization tensor with remapped IDs
                norm_tensor = torch.ones(256, dtype=torch.float32)
                for name, props in features.items():
                    if original_feature_mapping and name in original_feature_mapping and "normalization" in props:
                        feature_id = props["id"]
                        norm_value = props["normalization"]
                        norm_tensor[feature_id] = norm_value

                # Update the component's normalization tensor
                if hasattr(component, "update_normalization_tensor"):
                    component.update_normalization_tensor(norm_tensor)
