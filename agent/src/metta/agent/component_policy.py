import logging
from typing import Optional, Union

import gymnasium as gym
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn

from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.datastruct import duplicates
from metta.common.util.instantiate import instantiate

logger = logging.getLogger("component_policy")


class ComponentPolicy(nn.Module):
    def __init__(
        self,
        obs_space: Optional[Union[gym.spaces.Space, gym.spaces.Dict]] = None,
        obs_width: Optional[int] = None,
        obs_height: Optional[int] = None,
        action_space: Optional[gym.spaces.Space] = None,
        feature_normalizations: Optional[dict[int, float]] = None,
        device: Optional[str] = None,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__()

        # If no config provided, create a minimal policy (for backwards compatibility)
        if cfg is None:
            self.components = None
            self.clip_range = 0.0
            self.cum_action_max_params = None
            self.action_index_tensor = None
            return

        # Build components immediately, just like old MettaAgent did
        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        self.clip_range = self.cfg.clip_range

        # Validate and extract observation key
        if not (hasattr(self.cfg.observations, "obs_key") and self.cfg.observations.obs_key is not None):
            raise ValueError("Configuration missing required field 'observations.obs_key'")

        obs_key = self.cfg.observations.obs_key
        obs_shape = safe_get_from_obs_space(obs_space, obs_key, "shape")

        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": action_space,
            "feature_normalizations": feature_normalizations,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "obs_key": self.cfg.observations.obs_key,
            "obs_shape": obs_shape,
        }

        self.components = nn.ModuleDict()
        component_cfgs = self.cfg.components

        # First pass: instantiate all configured components
        for component_key in component_cfgs:
            component_name = str(component_key)
            comp_dict = dict(component_cfgs[component_key], **self.agent_attributes, name=component_name)
            self.components[component_name] = instantiate(comp_dict)

        # Setup components
        component = self.components["_value_"]
        self._setup_components(component)
        component = self.components["_action_"]
        self._setup_components(component)

        # Track components with memory
        self.components_with_memory = []
        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(f"Component {name} in ComponentPolicy was never setup.")
            if component.has_memory():
                self.components_with_memory.append(name)

        # Check for duplicate component names
        all_names = [c._name for c in self.components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        self.components = self.components.to(device)

        logger.info(f"ComponentPolicy components: {self.components}")

        # Initialize action conversion tensors (will be set by MettaAgent)
        self.cum_action_max_params = None
        self.action_index_tensor = None

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass of the ComponentPolicy - matches original MettaAgent forward() logic."""

        # Handle BPTT reshaping like the original
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

        self.components["_value_"](td)
        self.components["_action_"](td)

        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            output_td = output_td.reshape(batch_size, bptt_size)

        return output_td

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """Inference mode - sample actions and store them in td."""
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
        """Training mode - evaluate provided actions."""
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
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _setup_components(self, component):
        """Setup component connections - matching old MettaAgent logic."""
        if not hasattr(component, "_sources"):
            return

        source_components = {}
        for source in component._sources:
            source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def reset_memory(self) -> None:
        """Reset memory for all components that have memory."""
        if hasattr(self, "components_with_memory"):
            for name in self.components_with_memory:
                comp = self.components[name]
                if not hasattr(comp, "reset_memory"):
                    raise ValueError(
                        f"Component '{name}' listed in components_with_memory but has no reset_memory() method."
                        + " Perhaps an obsolete policy?"
                    )
                comp.reset_memory()

    def _apply_to_components(self, method_name, *args, **kwargs):
        """Apply a method to all components that have it."""
        if not self.components:
            return []

        results = []
        for _, component in self.components.items():
            if hasattr(component, method_name):
                method = getattr(component, method_name)
                if callable(method):
                    result = method(*args, **kwargs)
                    if result is not None:
                        results.append(result)
        return results

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0 and self.components:
            self._apply_to_components("clip_weights")
