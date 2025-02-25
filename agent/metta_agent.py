from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
import gymnasium as gym
import hydra
from omegaconf import OmegaConf

from sample_factory.utils.typing import ActionSpace, ObsSpace
from pufferlib.cleanrl import sample_logits
from pufferlib.environment import PufferEnv
import pufferlib


def make_policy(env: PufferEnv, cfg: OmegaConf):
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=[ 0 ],
            dtype=np.int32)
    })
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        _recursive_=False)

    agent.to(cfg.device)
    return agent

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    """
    A simple feature encoder that maps an input observation to a feature embedding φ.
    Adjust the architecture (e.g. CNN or MLP) based on your observation type.
    """
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        # Example for vector observations – use a CNN if using images.
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, feature_dim)

    def forward(self, x):
        # Flatten the input if necessary.
        if len(x.shape) == 5:
            x = x.reshape(-1, *x.shape[2:])
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        phi = self.fc2(x)
        return phi


class InverseDynamicsModel(nn.Module):
    """
    The inverse dynamics model g that takes a pair of embeddings (φ(sₜ), φ(sₜ₊₁))
    and outputs predictions over two components: action type and action parameter.
    """
    def __init__(self, feature_dim: int, action_type_dim: int, action_param_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Concatenate the two embeddings so the input is feature_dim * 2.
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        # Two heads: one for each action component.
        self.action_type_head = nn.Linear(hidden_dim, action_type_dim)
        self.action_param_head = nn.Linear(hidden_dim, action_param_dim)

    def forward(self, phi_s, phi_sp1):
        # Concatenate along the feature dimension.
        x = torch.cat([phi_s, phi_sp1], dim=1)
        x = self.relu(self.fc1(x))
        logits_type = self.action_type_head(x)
        logits_param = self.action_param_head(x)
        return logits_type, logits_param


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
        feature_dim=256,
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range

        agent_attributes = {
            'obs_shape': obs_shape,
            'clip_range': self.clip_range,
            'action_space': action_space,
            'grid_features': grid_features,
            'obs_key': cfg.observations.obs_key,
            'obs_input_shape': obs_space[cfg.observations.obs_key].shape[1:],
            'num_objects': obs_space[cfg.observations.obs_key].shape[2], # this is hardcoded for channel # at end of tuple
            'hidden_size': self.hidden_size,
            'core_num_layers': self.core_num_layers
        }

        if isinstance(action_space, pufferlib.spaces.Discrete):
            self._multi_discrete = False
            agent_attributes['action_type_size'] = action_space.n
        else:
            self._multi_discrete = True
            agent_attributes['action_type_size'] = action_space.nvec[0]
            agent_attributes['action_param_size'] = action_space.nvec[1]

        # self.observation_space = obs_space # for use with FeatureSetEncoder
        # self.global_features = global_features # for use with FeatureSetEncoder

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], **agent_attributes)
            self.components[component_cfg] = component

        component = self.components['_value_']
        self._setup_components(component)
        component = self.components['_action_type_']
        self._setup_components(component)
        if self._multi_discrete:
            component = self.components['_action_param_']
            self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup. It might not be accessible by other components.")

        self._total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")


        # Add a separate feature encoder for inverse dynamics.
        self.phi_encoder = FeatureEncoder(input_dim=int(np.prod(obs_shape)), feature_dim=feature_dim)
        # And an inverse dynamics model to predict actions from consecutive embeddings.
        self.inverse_dynamics_model = InverseDynamicsModel(feature_dim=feature_dim, action_type_dim=agent_attributes['action_type_size'], action_param_dim=agent_attributes['action_param_size'])


    def _setup_components(self, component):
        if component.input_source is not None:
            if isinstance(component.input_source, str):
                self._setup_components(self.components[component.input_source])
            elif isinstance(component.input_source, list):
                for input_source in component.input_source:
                    self._setup_components(self.components[input_source])

        if component.input_source is not None:
            if isinstance(component.input_source, str):
                component.setup(self.components[component.input_source])
            elif isinstance(component.input_source, list):
                input_source_components = {}
                for input_source in component.input_source:
                    input_source_components[input_source] = self.components[input_source]
                component.setup(input_source_components)
        else:
            component.setup()

    @property
    def lstm(self):
        return self.components["_core_"].net

    @property
    def total_params(self):
        return self._total_params

    def get_value(self, x, state=None):
        td = TensorDict({"x": x, "state": state})
        self.components["_value_"](td)
        return None, td["_value_"], None

    def get_action_and_value(self, x, state=None, action=None, e3b=None):
        td = TensorDict({"x": x})

        td["state"] = None
        if state is not None:
            state = torch.cat(state, dim=0)
            td["state"] = state.to(x.device)

        self.components["_value_"](td)
        self.components["_action_type_"](td)
        logits = td["_action_type_"]
        if self._multi_discrete:
            self.components["_action_param_"](td)
            logits = [logits, td["_action_param_"]]

        value = td["_value_"]
        state = td["state"]

        # Convert state back to tuple to pass back to trainer
        if state is not None:
            split_size = self.core_num_layers
            state = (state[:split_size], state[split_size:])

        phi = self.phi_encoder(x.to(next(self.phi_encoder.parameters()).device))
        e3b, intrinsic_reward = self._e3b_update(phi.detach(), e3b)

       # e3b, intrinsic_reward = self._e3b_update(td["_core_"].detach(), e3b)

        action, logprob, entropy = sample_logits(logits, action, False)
        return action, logprob, entropy, value, state, e3b, intrinsic_reward

    def forward(self, x, state=None, action=None, e3b=None):
        return self.get_action_and_value(x, state, action, e3b)

    def compute_inverse_dynamics_loss(self, current_obs, next_obs, action_type_targets, action_param_targets):
        # Compute embeddings for current and next observations.
        b, n, _, _, _ = current_obs.shape
        phi_current = self.phi_encoder(current_obs)
        phi_next = self.phi_encoder(next_obs)
        # Get predictions from your inverse dynamics model.
        logits_type, logits_param = self.inverse_dynamics_model(phi_current, phi_next)

        # Compute losses for each head.
        loss_type = F.cross_entropy(logits_type.reshape(b,n, -1), action_type_targets)
        loss_param = F.cross_entropy(logits_param.reshape(b,n, -1), action_param_targets.reshape(b*n))

        return loss_type + loss_param

    def _e3b_update(self, phi, e3b):
        intrinsic_reward = None
        if e3b is not None:
            # Calculate bonus (intrinsic reward)
            u = e3b @ phi.unsqueeze(2)  # C⁻¹φ
            intrinsic_reward = phi.unsqueeze(1) @ u  # φᵀC⁻¹φ

            # Sherman-Morrison update
            denominator = 1 + intrinsic_reward
            e3b = e3b - (u @ phi.unsqueeze(1) @ e3b) / denominator

            intrinsic_reward = intrinsic_reward.squeeze()
        return e3b, intrinsic_reward

    def l2_reg_loss(self) -> torch.Tensor:
        '''L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)

    def l2_init_loss(self) -> torch.Tensor:
        '''L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        '''Update interval set by l2_init_weight_update_interval. 0 means no updating.'''
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def clip_weights(self):
        '''Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global loss value or 0 to turn it off.'''
        if self.clip_range > 0:
            for component in self.components.values():
                component.clip_weights()

    def compute_effective_rank(self, delta: float = 0.01) -> List[dict]:
        '''Effective rank computation is off by default. Set effective_rank to True in the config to turn it on for a given component.'''
        effective_ranks = []
        for component in self.components.values():
            rank = component.effective_rank(delta)
            if rank is not None:
                effective_ranks.append(rank)
        return effective_ranks
