import logging
from typing import TYPE_CHECKING, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.rl.puffer_policy import PytorchAgent
from metta.common.util.instantiate import instantiate

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod



if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")



# =============================================================================
# POLICY BASE CLASS
# =============================================================================

class PolicyBase(nn.Module, ABC):
    """Base class for all policies with standardized interface."""

    def __init__(self, name: str = "BasePolicy"):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        agent: "MettaAgent",
        obs: Dict[str, torch.Tensor],
        state: Optional[PolicyState] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Execute policy forward pass."""
        pass


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class MettaAgent(nn.Module):
    """Clean and simplified MettaAgent implementation."""

    def __init__(self, components: nn.ModuleDict, config: DictConfig, policy: Optional[PolicyBase] = None, device = 'cpu'):
        super().__init__()
        self.components = components
        self.config = config
        self.device = device

        # Extract key configuration values
        self.hidden_size = self.config.components._core_.output_size
        self.core_num_layers = self.config.components._core_.nn_params.num_layers
        self.clip_range = self.config.clip_range

        # Set default policy if none provided
        if policy is None:
            policy = DefaultPolicy()
        self.set_policy(policy)

        # Move components to device and calculate parameters
        self.components = self.components.to(self.device)
        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"MettaAgent initialized with {self._total_params:,} parameters")

    def set_policy(self, policy: PolicyBase):
        """Set or change the agent's policy."""
        if not isinstance(policy, PolicyBase):
            raise TypeError("Policy must inherit from PolicyBase")
        self.policy = policy
        self.policy.agent = self

    def forward(self, obs: Dict[str, torch.Tensor], state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None) -> Tuple:
        """Forward pass through the policy."""
        if self.policy is None:
            raise RuntimeError("No policy set. Use set_policy() first.")
        return self.policy.forward(self, obs, state, action)

    # =============================================================================
    # ENVIRONMENT INITIALIZATION
    # =============================================================================

    def initialize_to_environment(self, features: dict[str, dict], action_names: list[str], action_max_params: list[int], device, is_training: bool = True):
        """Initialize the agent to the current environment."""
        self._initialize_observations(features, device, self.training)
        self.activate_actions(action_names, action_max_params, device)

    def _initialize_observations(self, features: dict[str, dict], device, is_training: bool):
        """Initialize observation features and handle feature remapping."""
        self.active_features = features
        self.device = device
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props}

        # Handle feature mapping for model reuse
        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            self._create_feature_remapping(features, is_training)

    def _create_feature_remapping(self, features: dict[str, dict], is_training: bool):
        """Create remapping for feature IDs when environment changes."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not is_training:
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            logger.info(f"Created feature remapping: {len(self.feature_id_remap)} remapped, {len(unknown_features)} unknown")
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to components."""
        if "_obs_" in self.components and hasattr(self.components["_obs_"], "update_feature_remapping"):
            remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)

            for new_id, original_id in self.feature_id_remap.items():
                remap_tensor[new_id] = original_id

            current_feature_ids = {props["id"] for props in features.values()}
            for feature_id in range(256):
                if feature_id not in self.feature_id_remap and feature_id not in current_feature_ids:
                    remap_tensor[feature_id] = unknown_id

            self.components["_obs_"].update_feature_remapping(remap_tensor)

        self._update_normalization_factors(features)

    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors for components."""
        for comp_name, component in self.components.items():
            if hasattr(component, "__class__") and "ObsAttrValNorm" in component.__class__.__name__:
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

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving."""
        return getattr(self, "original_feature_mapping", None)

    def restore_original_feature_mapping(self, mapping: dict[str, int]) -> None:
        """Restore feature mapping from saved state."""
        self.original_feature_mapping = mapping.copy()
        logger.info(f"Restored feature mapping with {len(mapping)} features")

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

    # =============================================================================
    # COMPONENT OPERATIONS
    # =============================================================================

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

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        results = {}
        for name, component in self.components.items():
            if hasattr(component, "compute_weight_metrics"):
                result = component.compute_weight_metrics(delta)
                if result is not None:
                    results[name] = result
        return list(results.values())


# =============================================================================
# DEFAULT POLICY IMPLEMENTATION
# =============================================================================

class DefaultPolicy(PolicyBase):
    """Default policy implementation that uses MettaAgent components."""

    def __init__(self):
        super().__init__(name="DefaultPolicy")
        self.agent = None

    def forward(self, agent: "MettaAgent", obs: Dict[str, torch.Tensor], state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None) -> Tuple:
        """Execute policy forward pass."""
        self.agent = agent

        # Extract observation tensor
        if isinstance(obs, dict):
            x = obs.get('x', obs.get('observation', next(iter(obs.values()))))
        else:
            x = obs

        # Initialize state if needed
        if state is None:
            state = PolicyState()

        # Prepare tensor dictionary for component processing
        td = {"x": x, "state": None}
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_h = state.lstm_h.to(x.device)
            lstm_c = state.lstm_c.to(x.device)
            td["state"] = torch.cat([lstm_h, lstm_c], dim=0)

        # Get value estimate
        if "_value_" in agent.components:
            agent.components["_value_"](td)
            value = td.get("_value_", torch.zeros(x.shape[0], 1, device=x.device))
        else:
            value = torch.zeros(x.shape[0], 1, device=x.device)

        if __debug__:
            assert_shape(value, ("BT", 1), "value")

        # Get action logits
        if "_action_" in agent.components:
            agent.components["_action_"](td)
            logits = td.get("_action_", torch.zeros(x.shape[0], 10, device=x.device))
        else:
            logits = torch.zeros(x.shape[0], 10, device=x.device)

        if __debug__:
            assert_shape(logits, ("BT", "A"), "logits")

        # Update LSTM state
        if "_core_" in agent.components and "state" in td and td["state"] is not None:
            split_size = agent.core_num_layers
            state.lstm_h = td["state"][:split_size]
            state.lstm_c = td["state"][split_size:]

        # Return appropriate output based on mode
        if action is None:
            return self._forward_inference(value, logits)
        else:
            return self._forward_training(value, logits, action)

    def _forward_inference(self, value: torch.Tensor, logits: torch.Tensor) -> Tuple:
        """Forward pass for inference - sample new actions."""
        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)
        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_action")

        return action, action_log_prob, entropy, value, log_probs

    def _forward_training(self, value: torch.Tensor, logits: torch.Tensor, action: torch.Tensor) -> Tuple:
        """Forward pass for training - evaluate provided actions."""
        if __debug__:
            assert_shape(value, ("BT", 1), "training_value")
            assert_shape(logits, ("BT", "A"), "training_logits")
            assert_shape(action, ("B", "T", 2), "training_action")

        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)
        action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()

        cumulative_sum = self.agent.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        return action_logit_indices

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.agent.action_index_tensor[action_logit_index]


# =============================================================================
# AGENT BUILDER
# =============================================================================

class MettaAgentBuilder:
    """Simplified builder for MettaAgent instances."""

    def __init__(self, obs_space: Union[gym.spaces.Space, gym.spaces.Dict], obs_width: int, obs_height: int,
                 action_space: gym.spaces.Space, feature_normalizations: dict[int, float], device: str, **cfg):
        self.cfg = OmegaConf.create(cfg)

        # Extract key configuration
        self.hidden_size = self.cfg.components._core_.output_size
        self.core_num_layers = self.cfg.components._core_.nn_params.num_layers
        self.clip_range = self.cfg.clip_range
        self.device = device

        # Validate and extract observation key
        if not (hasattr(self.cfg.observations, "obs_key") and self.cfg.observations.obs_key is not None):
            raise ValueError("Configuration missing required field 'observations.obs_key'")

        obs_key = self.cfg.observations.obs_key
        obs_shape = safe_get_from_obs_space(obs_space, obs_key, "shape")

        # Prepare agent attributes for component instantiation
        self.agent_attributes = {
            "clip_range": self.clip_range,
            "action_space": action_space,
            "feature_normalizations": feature_normalizations,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "obs_key": obs_key,
            "obs_shape": obs_shape,
            "hidden_size": self.hidden_size,
            "core_num_layers": self.core_num_layers,
        }

        # Build components
        self.components = self._build_components()
        self.components = self.components.to(device)

        logger.info(f"Builder initialized with {len(self.components)} components")

    def _build_components(self) -> nn.ModuleDict:
        """Build all agent components."""
        components = nn.ModuleDict()
        component_cfgs = self.cfg.components

        # Instantiate all components
        for component_key in component_cfgs:
            component_name = str(component_key)
            comp_dict = dict(component_cfgs[component_key], **self.agent_attributes, name=component_name)
            components[component_name] = instantiate(comp_dict)

        # Setup components with dependencies
        if "_value_" in components:
            self._setup_component_tree(components["_value_"], components)
        if "_action_" in components:
            self._setup_component_tree(components["_action_"], components)

        # Validate all components are ready
        for name, component in components.items():
            if not getattr(component, "ready", False):
                raise RuntimeError(f"Component {name} was never setup properly")

        return components

    def _setup_component_tree(self, component, all_components):
        """Recursively setup component dependencies."""
        if hasattr(component, '_sources') and component._sources is not None:
            for source in component._sources:
                source_name = source['name']
                if source_name in all_components:
                    self._setup_component_tree(all_components[source_name], all_components)

        # Setup current component
        source_components = None
        if hasattr(component, '_sources') and component._sources is not None:
            source_components = {source['name']: all_components[source['name']]
                               for source in component._sources if source['name'] in all_components}

        if hasattr(component, 'setup'):
            component.setup(source_components)

    def build(self, policy: Optional[PolicyBase] = None) -> MettaAgent:
        """Build the final MettaAgent instance."""
        try:
            if policy is None:
                policy = DefaultPolicy()
            elif not isinstance(policy, PolicyBase):
                if hasattr(policy, '_target_'):
                    policy = instantiate(policy)
                else:
                    raise TypeError("Policy must be PolicyBase instance or have '_target_' field")

            agent = MettaAgent(components=self.components, config=self.cfg, policy=policy, device=self.device)
            logger.info(f"Built MettaAgent with {len(self.components)} components")
            return agent

        except Exception as e:
            logger.error(f"Failed to build MettaAgent: {e}")
            raise


# =============================================================================
# DISTRIBUTED WRAPPER
# =============================================================================

class DistributedMettaAgent(DistributedDataParallel):
    """Distributed wrapper for MettaAgent."""

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

    def initialize_to_environment(self, features: dict[str, dict], action_names: list[str],
                                action_max_params: list[int], device: torch.device, is_training: bool = True) -> None:
        return self.module.initialize_to_environment(features, action_names, action_max_params, device)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def make_policy(env: "MettaGridEnv", cfg: DictConfig) -> "MettaAgent":
    """Factory function to create MettaAgent from environment and config."""
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
    })

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    logger.info(f"Agent Config: {OmegaConf.create(agent_cfg)}")

    builder = MettaAgentBuilder(
        obs_space=obs_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        action_space=env.single_action_space,
        feature_normalizations=env.feature_normalizations,
        device=cfg.device,
        **agent_cfg,
    )

    return builder.build()




PolicyAgent = MettaAgent | DistributedMettaAgent | PytorchAgent
