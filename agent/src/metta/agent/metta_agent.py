import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import NonTensorData, TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.pytorch.agent_mapper import agent_classes
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.datastruct import duplicates
from metta.common.util.instantiate import instantiate
from metta.rl.env_config import EnvConfig
from metta.rl.puffer_policy import PytorchAgent

if TYPE_CHECKING:
    from metta.mettagrid import MettaGridEnv

logger = logging.getLogger("metta_agent")


def make_policy(env: "MettaGridEnv", env_cfg: EnvConfig, agent_cfg: DictConfig) -> "MettaAgent":
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Check if agent_cfg specifies a pytorch agent type
    if "agent_type" in agent_cfg and agent_cfg.agent_type in agent_classes:
        AgentClass = agent_classes[agent_cfg.agent_type]
        agent = AgentClass(env=env)
        logger.info(f"Using Pytorch Policy: {agent} (type: {agent_cfg.agent_type})")
        return agent

    # For backward compatibility with YAML configs (to be removed in future PR)
    dict_agent_cfg: dict = OmegaConf.to_container(agent_cfg, resolve=True)

    return MettaAgent(
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        device=env_cfg.device,
        **dict_agent_cfg,
    )


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

        self.components_with_memory = []
        for name, component in self.components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(
                    f"Component {name} in MettaAgent was never setup. It might not be accessible by other components."
                )
            if component.has_memory():
                self.components_with_memory.append(name)

        # check for duplicate component names
        all_names = [c._name for c in self.components.values() if hasattr(c, "_name")]
        if duplicate_names := duplicates(all_names):
            raise ValueError(f"Duplicate component names found: {duplicate_names}")

        self.components = self.components.to(device)

        self._total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")

    def reset_memory(self):
        for name in self.components_with_memory:
            self.components[name].reset_memory()

    def get_memory(self):
        memory = {}
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

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
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
        # Use PyTorch's built-in training mode detection
        self._initialize_observations(features, device)
        self.activate_actions(action_names, action_max_params, device)

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

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Run this at the beginning of training."""
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
        self.components["_action_embeds_"].activate_actions(full_action_names, self.device)

        # Create action_index tensor
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)
        logger.info(f"Agent actions initialized with: {self.active_actions}")

    @property
    def total_params(self):
        return self._total_params

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """
        Forward pass for inference mode - softmaxes action logits then samples them and outputs new actions.
        """
        value = td["_value_"]
        logits = td["_action_"]

        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        # Sample actions
        action_logit_index, action_log_prob, _, full_log_probs = sample_actions(logits)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")

        # Convert logit index to action
        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_output_action")

        td["actions"] = action
        td["act_log_prob"] = action_log_prob
        td["values"] = value.flatten()
        td["full_log_probs"] = full_log_probs

        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """
        Forward pass for training mode - evaluates the policy on provided actions from rollout.
        Finds the action logprobs and calculates entropy.
        """
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

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        """
        Forward pass of the MettaAgent - delegates to appropriate specialized method.

        Args:
            td: A TensorDict containing at least "env_obs". In training, it should also contain the keys that are
            specified in the experience buffer spec function also defined in this class.
            action: Optional action tensor for BPTT (training mode).

        Returns:
            A TensorDict containing the model's output.
            - In inference mode, this contains data to be stored in the experience buffer.
            - In training mode, this contains data for loss calculation.
        """
        if td.batch_dims > 1:
            B, TT = td.batch_size
            td.set("bptt", NonTensorData(TT))
            td.set("batch", NonTensorData(B))
            td = td.reshape(B * TT)
        else:
            td.set("bptt", NonTensorData(1))
            td.set("batch", NonTensorData(td.batch_size.numel()))

        # Forward pass through value network. This will also run the core network.
        self.components["_value_"](td)

        # Forward pass through action network. This will reuse the core network's output.
        self.components["_action_"](td)

        # TODO: future work could allow losses to decide which leaf nodes to run eg for reconstruction loss

        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            output_td = output_td.reshape(B, TT)

        return output_td

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


PolicyAgent = MettaAgent | DistributedMettaAgent | PytorchAgent
