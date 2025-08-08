import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.pytorch.agent_mapper import agent_classes
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.datastruct import duplicates
from metta.common.util.instantiate import instantiate


from typing import Dict


if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class ComponentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = None
        self.clip_range = 0.0

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass of the ComponentPolicy - matches original MettaAgent forward() logic."""

        if self.components is None:
            raise ValueError("No components found. Ensure components are added in YAML.")

        # Handle BPTT reshaping like the original
        td.bptt = 1
        td.batch = td.batch_size.numel()
        if td.batch_dims > 1:
            B = td.batch_size[0]
            TT = td.batch_size[1]
            td = td.reshape(td.batch_size.numel())  # flatten to BT
            td.bptt = TT
            td.batch = B

        # Run value head (also runs core network if present)
        self.components["_value_"](td)

        # Run action head (reuses core network output)
        self.components["_action_"](td)

        # Select forward pass type
        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            output_td = output_td.reshape(B, TT)

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

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")



class MettaAgent(nn.Module):
    """Clean and simplified MettaAgent implementation."""

    def __init__(
        self,
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        obs_width: int,
        obs_height: int,
        action_space: gym.spaces.Space,
        feature_normalizations: dict[int, float],
        device: str,
        cfg,
        policy
    ):
        super().__init__()
        self.cfg = cfg
        self.policy = policy
        self.device = device

        self.obs_space = obs_space
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.action_space = action_space
        self.feature_normalizations = feature_normalizations


        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")



    def set_policy(self, policy):
        """Set the agent's policy."""
        self.policy = policy
        self.policy.agent = self
        self.policy.to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor], state = None, action: Optional[torch.Tensor] = None) -> Tuple:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")
        if isinstance(self.policy, ComponentPolicy):

            return self.policy.forward(obs, action)

        logger.info(f"Obervation: {obs['env_obs'].shape}")

        return self.policy(obs['env_obs'], state, action)


    def initialize_to_environment(self, features: dict[str, dict], action_names: list[str], action_max_params: list[int], device, is_training: bool = True):
        """Initialize the agent to the current environment."""
        self._initialize_observations(features, device)

        if isinstance(self.policy, ComponentPolicy):
            self.activate_policy()
            self.activate_actions(action_names, action_max_params, device)
        else:
            self.policy.initialize_to_environment(features, action_names, action_max_params, device, is_training)



    def activate_policy(self):
        # self.hidden_size = self.cfg.components._core_.output_size
        # self.core_num_layers = self.cfg.components._core_.nn_params.num_layers
        self.clip_range = self.cfg.clip_range

        # Validate and extract observation key
        if not (hasattr(self.cfg.observations, "obs_key") and self.cfg.observations.obs_key is not None):
            raise ValueError("Configuration missing required field 'observations.obs_key'")

        obs_key = self.cfg.observations.obs_key
        obs_shape = safe_get_from_obs_space(self.obs_space, obs_key, "shape")


        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": self.action_space,
            "feature_normalizations": self.feature_normalizations,
            "obs_width": self.obs_width,
            "obs_height": self.obs_height,
            "obs_key": self.cfg.observations.obs_key,
            "obs_shape": obs_shape,
        }


        self.components = nn.ModuleDict()
        # Keep component configs as DictConfig to support both dict and attribute access
        component_cfgs = self.cfg.components

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

        self.components_with_memory = []
        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(
                    f"Component {name} in MettaAgent was never setup. It might not be accessible by other components."
                )
            if component.has_memory():
                self.components_with_memory.append(name)

        # logger.info(f"Components with memory: {self.components_with_memory}")

        # check for duplicate component names
        all_names = [c._name for c in self.components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        self.components = self.components.to(self.device)

        logger.info(f"MettaAgent components: {self.components}")

        self.policy.components = self.components
        self.policy.clip_range = self.clip_range


    def reset_memory(self):
        if isinstance(self.policy, ComponentPolicy):
            for name in self.components_with_memory:
                self.components[name].reset_memory()

    def get_memory(self):
        memory = {}
        if isinstance(self.policy, ComponentPolicy):
            for name in self.components_with_memory:
                memory[name] = self.components[name].get_memory()
        return memory

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )



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




    def _initialize_observations(self, features: dict[str, dict], device):
        """Initialize observation features by storing the feature mapping."""
        self.active_features = features
        self.device = device

        # Create quick lookup mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        # Store original feature mapping on first initialization
        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            # Create remapping for subsequent initializations
            self._create_feature_remapping(features)



    def _create_feature_remapping(self, features: dict[str, dict]):
        """Create a remapping dictionary to translate new feature IDs to original ones."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                # Remap known features to their original IDs
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not self.training:
                # In eval mode, map unknown features to UNKNOWN_FEATURE_ID
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                # In training mode, learn new features
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            logger.info(
                f"Created feature remapping: {len(self.feature_id_remap)} remapped, {len(unknown_features)} unknown"
            )
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)



    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to observation component and update normalizations."""
        # Update observation component if it supports remapping
        if "_obs_" in self.components and hasattr(self.components["_obs_"], "update_feature_remapping"):
            # Build complete remapping tensor
            remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)

            # Apply explicit remappings
            for new_id, original_id in self.feature_id_remap.items():
                remap_tensor[new_id] = original_id

            # Map unused feature IDs to UNKNOWN
            current_feature_ids = {props["id"] for props in features.values()}
            for feature_id in range(256):
                if feature_id not in self.feature_id_remap and feature_id not in current_feature_ids:
                    remap_tensor[feature_id] = unknown_id

            self.components["_obs_"].update_feature_remapping(remap_tensor)

        # Update normalization factors
        self._update_normalization_factors(features)


    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors for components after feature remapping."""
        # Update ObsAttrValNorm components if they exist
        for comp_name, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
                logger.info(f"Updating feature normalizations for {comp_name}")

                # Create normalization tensor with remapped IDs
                norm_tensor = torch.ones(256, dtype=torch.float32)
                for name, props in features.items():
                    if name in self.original_feature_mapping and "normalization" in props:
                        original_id = self.original_feature_mapping[name]
                        norm_tensor[original_id] = props["normalization"]

                component.register_buffer("_norm_factors", norm_tensor)


    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Initialize action space for the agent."""
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for efficient action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        # Build full action names and activate embeddings
        full_action_names = []
        for action_name, max_param in self.active_actions:
            for i in range(max_param + 1):
                full_action_names.append(f"{action_name}_{i}")

        if "_action_embeds_" in self.components:
            self.components["_action_embeds_"].activate_actions(full_action_names, self.device)

        # Create action index tensor for conversions
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"Actions initialized: {self.active_actions}")


        # Activate policy attributes
        self.policy.action_index_tensor = self.action_index_tensor
        self.policy.cum_action_max_params = self.cum_action_max_params


    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving in metadata."""
        return getattr(self, "original_feature_mapping", None)

    def restore_original_feature_mapping(self, mapping: dict[str, int]) -> None:
        """Restore the original feature mapping from metadata.

        This should be called after loading a model from checkpoint but before
        calling initialize_to_environment.
        """
        # Make a copy to avoid shared state between agents
        self.original_feature_mapping = mapping.copy()
        logger.info(f"Restored original feature mapping with {len(mapping)} features from metadata")

    @property
    def lstm(self):
        """Access to LSTM component."""
        if "_core_" in self.components and hasattr(self.components["_core_"], "_net"):
            return self.components["_core_"]._net
        return None

    @property
    def total_params(self):
        """Total number of parameters."""
        return getattr(self, '_total_params', sum(p.numel() for p in self.parameters() if p.requires_grad))


    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """Apply a method to all components that have it."""
        results = []
        for name, component in self.components.items():
            if hasattr(component, method_name):
                method = getattr(component, method_name)
                if callable(method):
                    result = method(*args, **kwargs)
                    if result is not None:
                        results.append(result)
        return results

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss."""
        losses = self._apply_to_components("l2_init_loss")
        return torch.sum(torch.stack(losses)) if losses else torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copies."""
        self._apply_to_components("update_l2_init_weight_copy")


    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        results = {}
        for name, component in self.components.items():
            if hasattr(component, "compute_weight_metrics"):
                result = component.compute_weight_metrics(delta)
                if result is not None:
                    results[name] = result
        return list(results.values())


class DistributedMettaAgent(DistributedDataParallel):
    """
    Because this class passes through __getattr__ to its self.module, it implements everything
    MettaAgent does. We only have a need for this class because using the DistributedDataParallel wrapper
    returns an object of almost the same interface: you need to call .module to get the wrapped agent.
    """

    module: "MettaAgent"

    def __init__(self, agent: "MettaAgent", device: torch.device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")

        # This maintains the same interface as the input MettaAgent
        layers_converted_agent: "MettaAgent" = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)  # type: ignore

        # Pass device_ids for GPU, but not for CPU
        if device.type == "cpu":
            super().__init__(module=layers_converted_agent)
        else:
            super().__init__(module=layers_converted_agent, device_ids=[device], output_device=device)

    def __getattr__(self, name: str) -> Any:
        # First try DistributedDataParallel's __getattr__, then self.module's (MettaAgent's)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


PolicyAgent = MettaAgent | DistributedMettaAgent
