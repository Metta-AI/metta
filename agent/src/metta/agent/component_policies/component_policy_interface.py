import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.agent_config import ComponentPolicyConfig
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.datastruct import duplicates

logger = logging.getLogger(__name__)


class ComponentPolicyInterface(nn.Module, ABC):
    """
    Abstract base class for component-based policies.
    """

    def __init__(
        self,
        config: ComponentPolicyConfig,
    ):
        super().__init__()

        self.config = config
        self.clip_range = 0

        # Device setup
        self.device = (
            torch.device(config.device)
            if config.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Validate and extract observation key
        obs_key = "grid_obs"
        obs_shape = (
            safe_get_from_obs_space(config.obs_space, obs_key, "shape") if config.obs_space is not None else None
        )

        # Create agent attributes dict
        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": config.action_space,
            "feature_normalizations": config.feature_normalizations,
            "obs_width": config.obs_width,
            "obs_height": config.obs_height,
            "obs_key": obs_key,
            "obs_shape": obs_shape,
        }

        # Action space handling
        if config.action_space is not None and hasattr(config.action_space, "nvec"):
            action_nvec = config.action_space.nvec
        else:
            action_nvec = [100]  # default

        self.action_nvec = action_nvec
        self.components = nn.ModuleDict()

        # Build components using the abstract method
        components = self._build_components()

        # Add all components to the ModuleDict
        self.components.update(components)

        # Setup components
        self._setup_components(self.components["_value_"])
        self._setup_components(self.components["_action_"])

        # Track components with memory
        self.components_with_memory = []
        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(f"Component {name} in {self.__class__.__name__} policy was never setup.")
            if hasattr(component, "has_memory") and component.has_memory():
                self.components_with_memory.append(name)

        # Check for duplicate component names
        all_names = [c._name for c in self.components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        # Move to device
        self.components = self.components.to(self.device)

        logger.info(f"{self.__class__.__name__} policy components: {self.components}")

        # Initialize action conversion tensors (will be set by MettaAgent)
        self.cum_action_max_params = None
        self.action_index_tensor = None

        logger.info(f"{self.__class__.__name__} policy initialized with components: {list(self.components.keys())}")

    @abstractmethod
    def _build_components(self) -> dict:
        """Build the component dictionary. Must be implemented by subclasses."""
        pass

    def _setup_components(self, component):
        """Setup component connections."""
        # Skip if already setup
        if getattr(component, "ready", False):
            return

        # Recursively setup all source components first
        if hasattr(component, "_sources") and component._sources is not None:
            for source in component._sources:
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

    def reset_memory(self) -> None:
        """Reset memory for all components that have memory."""
        for name in self.components_with_memory:
            comp = self.components[name]
            if not hasattr(comp, "reset_memory"):
                raise ValueError(
                    f"Component '{name}' listed in components_with_memory but has no reset_memory() method."
                    + " Perhaps an obsolete policy?"
                )
            comp.reset_memory()

    def get_memory(self) -> dict:
        """Get memory state from all components that have memory."""
        memory = {}
        for name in self.components_with_memory:
            if hasattr(self.components[name], "get_memory"):
                memory[name] = self.components[name].get_memory()
        return memory

    # ============================================================================
    # Weight/Training Utility Methods
    # ============================================================================

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss for all components."""
        losses = self._apply_to_components("l2_init_loss")
        return torch.sum(torch.stack(losses)) if losses else torch.tensor(0.0, dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies for all components."""
        self._apply_to_components("update_l2_init_weight_copy")

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

    # ============================================================================
    # Helper Methods and Properties
    # ============================================================================

    def _apply_to_components(self, method_name, *args, **kwargs):
        """Apply a method to all components that have it."""
        results = []
        for _, component in self.components.items():
            if hasattr(component, method_name):
                method = getattr(component, method_name)
                if callable(method):
                    result = method(*args, **kwargs)
                    if result is not None:
                        results.append(result)
        return results

    @property
    def lstm(self):
        """Access to LSTM component if it exists."""
        if "_core_" in self.components and hasattr(self.components["_core_"], "_net"):
            return self.components["_core_"]._net
        elif "_core_" in self.components:
            return self.components["_core_"]
        return None

    @property
    def hidden_size(self):
        """Get hidden size from LSTM component."""
        if "_core_" in self.components:
            return getattr(self.components["_core_"], "hidden_size", None)
        return None

    @property
    def num_lstm_layers(self):
        """Get number of LSTM layers."""
        if "_core_" in self.components:
            return getattr(self.components["_core_"], "num_layers", None)
        return None
