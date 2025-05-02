import logging
from typing import List, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pufferlib.environment import PufferEnv
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import sample_logits

logger = logging.getLogger("metta_agent")


def make_policy(env: PufferEnv, cfg: ListConfig | DictConfig):
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )
    return hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        device=cfg.device,
        _recursive_=False,
    )


class DistributedMettaAgent(DistributedDataParallel):
    def __init__(self, agent, device):
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
        action_space: gym.spaces.Space,
        grid_features: List[str],
        device: str,
        **cfg,
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range

        agent_attributes = {
            "obs_shape": obs_shape,
            "clip_range": self.clip_range,
            "action_space": action_space,
            "grid_features": grid_features,
            "obs_key": cfg.observations.obs_key,
            "obs_input_shape": obs_space[cfg.observations.obs_key].shape[1:],
            "num_objects": obs_space[cfg.observations.obs_key].shape[
                2
            ],  # this is hardcoded for channel # at end of tuple
            "hidden_size": self.hidden_size,
            "core_num_layers": self.core_num_layers,
        }
        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]["name"] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], **agent_attributes)
            self.components[component_cfg] = component

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

    def _setup_components(self, component):
        """_sources is a list of dicts albeit many layers simply have one element.
        It must always have a "name" and that name should be the same as the relevant key in self.components.
        source_components is a dict of components that are sources for the current component. The keys
        are the names of the source components."""
        # recursively setup all source components
        if component._sources is not None:
            for source in component._sources:
                self._setup_components(self.components[source["name"]])

        # setup the current component and pass in the source components
        source_components = None
        if component._sources is not None:
            source_components = {}
            for source in component._sources:
                source_components[source["name"]] = self.components[source["name"]]
        component.setup(source_components)

    def activate_actions(self, action_names, action_max_params, device):
        """Run this at the beginning of training."""
        self.device = device
        self.actions_max_params = action_max_params
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(torch.tensor([0] + action_max_params, device=self.device), dim=0)

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

        self.action_index_tensor = torch.tensor(action_index, device=self.device)
        logger.info(f"Agent actions activated with: {self.active_actions}")

    @property
    def lstm(self):
        return self.components["_core_"]._net

    @property
    def total_params(self):
        return self._total_params

    def forward(self, x, state: PolicyState, action=None):
        td = TensorDict(
            {
                "x": x,
                "state": None,
            }
        )

        if state.lstm_h is not None:
            td["state"] = torch.cat([state.lstm_h, state.lstm_c], dim=0).to(x.device)

        self.components["_value_"](td)
        value = td["_value_"]
        # state = td["state"]
        self.components["_action_"](td)
        logits = td["_action_"]

        split_size = self.core_num_layers
        state.lstm_h = td["state"][:split_size]
        state.lstm_c = td["state"][split_size:]

        action_logit_index = self._convert_action_to_logit_index(action) if action is not None else None
        action_logit_index, logprob_act, entropy, log_sftmx_logits = sample_logits(logits, action_logit_index)
        if action is None:
            action = self._convert_logit_index_to_action(action_logit_index, td)

        return action, logprob_act, entropy, value, log_sftmx_logits

    def _convert_action_to_logit_index(self, action):
        """Convert action pairs (action_type, action_param) to single discrete action logit indices using vectorized
        operations"""
        action = action.reshape(-1, 2)

        # Extract action components
        action_type_numbers = action[:, 0].long()
        action_params = action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]

        # Vectorized addition
        action_logit_index = action_type_numbers + cumulative_sum + action_params

        return action_logit_index.reshape(-1, 1)

    def _convert_logit_index_to_action(self, action_logit_index, td):
        """Convert logit indices back to action pairs using tensor indexing"""
        # direct tensor indexing on precomputed action_index_tensor
        return self.action_index_tensor[action_logit_index.reshape(-1)]

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust
        it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval. 0 means no updating."""
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def clip_weights(self):
        """Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value
        effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global
        loss value or 0 to turn it off."""
        if self.clip_range > 0:
            for component in self.components.values():
                component.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        """Compute weight metrics for all components that have weights enabled for analysis.
        Returns a list of metric dictionaries, one per component. Set analyze_weights to True in the config to turn it
        on for a given component."""
        weight_metrics = []
        for component in self.components.values():
            metrics = component.compute_weight_metrics(delta)
            if metrics is not None:
                weight_metrics.append(metrics)
        return weight_metrics
