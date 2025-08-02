import logging
from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple

import gymnasium as gym
from omegaconf import OmegaConf, DictConfig
from torch import nn
import torch

from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.agent.util.safe_get import safe_get_from_obs_space
from metta.common.util.instantiate import instantiate
from metta.agent.policy_base import PolicyBase
from metta.agent.policy_state import PolicyState
from metta.agent.metta_agent import MettaAgent

import numpy as np



logger = logging.getLogger("metta_agent_builder")


def make_policy(env: "MettaGridEnv", cfg: DictConfig) -> MettaAgent:

    """Factory function to create MettaAgent from environment and config."""
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
    })

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    logger.info(f"Agent Config: {OmegaConf.create(agent_cfg)}")

    logger.info(f"Feature Normalizations: {env.feature_normalizations}")

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

