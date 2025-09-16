import logging
from abc import abstractmethod
from typing import Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.lib.metta_layer import LayerBase
from metta.agent.policy import Policy
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.collections import duplicates
from metta.rl.training.training_environment import EnvironmentMetaData

logger = logging.getLogger(__name__)


class ComponentPolicy(nn.Module, Policy):
    """
    Abstract base class for component-based policies.
    Subclasses must override _build_components() to define their architecture.
    """

    def __init__(self, env_metadata: EnvironmentMetaData):
        super().__init__()

        # Extract observation shape (always uses "grid_obs" key)
        self._obs_key = "grid_obs"
        self._obs_shape = safe_get_from_obs_space(env_metadata.observation_space, self._obs_key, "shape")

        self._components = self._build_components(env_metadata)

        # register with pytorch
        self._components_module = nn.ModuleDict(self._components)

        # Setup components by triggering leaf node setup
        [self._setup_component(self._components[c]) for c in self._output_heads()]

        for c in self._components.keys():
            if not getattr(self._components[c], "ready", False):
                raise RuntimeError(f"Component {c} in {self.__class__.__name__} policy was never setup.")

        # Check for duplicate component names
        all_names = [c._name for c in self._components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        # Compute action tensors for efficient indexing
        self._cum_action_max_params = torch.cumsum(
            torch.tensor([0] + env_metadata.max_action_args, device=self.device, dtype=torch.long), dim=0
        )
        self._action_index_tensor = torch.tensor(
            [[idx, j] for idx, max_param in enumerate(env_metadata.max_action_args) for j in range(max_param + 1)],
            device=self.device,
            dtype=torch.int32,
        )

        logger.info(f"{self.__class__.__name__} policy initialized with components: {list(self._components.keys())}")

    def __repr__(self):
        return f"{self.__class__.__name__} policy with components: {list(self._components.keys())}"

    @abstractmethod
    def _build_components(self, env_metadata: EnvironmentMetaData) -> Dict[str, LayerBase]: ...

    @abstractmethod
    def _output_heads(self) -> list[str]: ...

    def _setup_component(self, component: LayerBase):
        """Setup component connections."""
        if component.ready:
            return

        # Recursively setup all source components first
        [self._setup_component(self._components[c]) for c in component._source_component_names]
        component.setup({c: self._components[c] for c in component._source_component_names})

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
        self._components["_value_"](td)
        self._components["_action_"](td)

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
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self._cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self._action_index_tensor[action_logit_index]

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize components to the environment with the given action names."""
        if "_action_embeds_" in self._components:
            self._components["_action_embeds_"].initialize_to_environment(full_action_names, device)

    # ============================================================================
    # Memory-related Methods
    # ============================================================================

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def on_new_training_run(self):
        [component.on_new_training_run() for component in self._components.values()]

    def on_rollout_start(self):
        [component.on_rollout_start() for component in self._components.values()]

    def on_train_mb_start(self):
        [component.on_train_mb_start() for component in self._components.values()]

    def on_eval_start(self):
        [component.on_eval_start() for component in self._components.values()]

    # ============================================================================
    # Weight/Training Utility Methods
    # ============================================================================

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for all components."""
        results = {}
        for name, component in self._components.items():
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
        if "_obs_" in self._components:
            obs_component = self._components["_obs_"]
            if hasattr(obs_component, "update_feature_remapping"):
                obs_component.update_feature_remapping(remap_tensor)

    def update_normalization_factors(self, features: dict[str, dict], original_feature_mapping: dict[str, int] | None):
        """Update normalization factors for ObsAttrValNorm components after feature remapping."""
        for component in self._components.values():
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

    def clip_weights(self):
        """Clip weights in all components that support it."""
        for component in self._components.values():
            if hasattr(component, "clip_weights"):
                component.clip_weights()

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss across all components."""
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        for component in self._components.values():
            if hasattr(component, "l2_init_loss"):
                total_loss += component.l2_init_loss()
        return total_loss

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, device: torch.device):
        [c.to(device) for c in self._components.values()]
        self._action_index_tensor = self._action_index_tensor.to(device)
        self._cum_action_max_params = self._cum_action_max_params.to(device)
