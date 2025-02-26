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
        feature_dim=128,
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)

        self.hidden_size = cfg.components._core_.output_size
        self.core_num_layers = cfg.components._core_.nn_params.num_layers
        self.clip_range = cfg.clip_range
        self.phi_previous = None  # Optional: if you wish to store previous embedding

        agent_attributes = {
            'obs_shape': obs_shape,
            'clip_range': self.clip_range,
            'action_space': action_space,
            'grid_features': grid_features,
            'obs_key': cfg.observations.obs_key,
            'obs_input_shape': obs_space[cfg.observations.obs_key].shape[1:],
            'num_objects': obs_space[cfg.observations.obs_key].shape[2],
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

        self.components = nn.ModuleDict()
        component_cfgs = OmegaConf.to_container(cfg.components, resolve=True)
        for component_cfg in component_cfgs.keys():
            component_cfgs[component_cfg]['name'] = component_cfg
            component = hydra.utils.instantiate(component_cfgs[component_cfg], **agent_attributes)
            self.components[component_cfg] = component

        # Setup necessary components.
        for name in ["_value_", "_action_type_"]:
            self._setup_components(self.components[name])
        if self._multi_discrete:
            self._setup_components(self.components["_action_param_"])

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup.")

        self._total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")

        # In the 'policy enc' variant, we reuse the shared encoder.
        # We do not create a separate phi_encoder here.
        # Instead, we will use the output of our shared pipeline ("encoded_obs")
        # as computed by get_shared_embedding().
        self.inverse_dynamics_model = InverseDynamicsModel(
            feature_dim=feature_dim,
            action_type_dim=agent_attributes['action_type_size'],
            action_param_dim=agent_attributes.get('action_param_size', 0),
            hidden_dim=256
        )
        # Optionally, you might add an optimizer for the inverse dynamics model if desired.

    def _setup_components(self, component):
        if component.input_source is not None:
            if isinstance(component.input_source, str):
                self._setup_components(self.components[component.input_source])
            elif isinstance(component.input_source, list):
                for src in component.input_source:
                    self._setup_components(self.components[src])
        if component.input_source is not None:
            if isinstance(component.input_source, str):
                component.setup(self.components[component.input_source])
            elif isinstance(component.input_source, list):
                src_components = {src: self.components[src] for src in component.input_source}
                component.setup(src_components)
        else:
            component.setup()

    @property
    def lstm(self):
        return self.components["_core_"].net

    @property
    def total_params(self):
        return self._total_params

    def get_shared_embedding(self, x):
        """
        Computes the shared embedding from the policy network.
        For example, we run x through _obs_, obs_normalizer, cnn1, cnn2, obs_flattener, fc1, and encoded_obs.
        Adjust these calls if your pipeline is different.
        """
        td = TensorDict({"x": x})
        self.components["_obs_"](td)
        self.components["obs_normalizer"](td)
        self.components["cnn1"](td)
        self.components["cnn2"](td)
        self.components["obs_flattener"](td)
        self.components["fc1"](td)
        self.components["encoded_obs"](td)
        return td["encoded_obs"]

    def get_value(self, x, state=None):
        td = TensorDict({"x": x, "state": state})
        self.components["_value_"](td)
        return None, td["_value_"], None

    def get_action_and_value(self, x, state=None, action=None, e3b=None):
        # Determine device once and stick to it
        device = x.device

        td = TensorDict({"x": x})
        td["state"] = None
        if state is not None:
            state = torch.cat(state, dim=0)
            td["state"] = state.to(device)  # Ensure state is on the same device

        self.components["_value_"](td)
        self.components["_action_type_"](td)
        logits = td["_action_type_"]
        if self._multi_discrete:
            self.components["_action_param_"](td)
            logits = [logits, td["_action_param_"]]
        # Safety check for NaN or inf values
        for i, l in enumerate(logits):
            if torch.isnan(l).any() or torch.isinf(l).any():
                print(f"Warning: NaN or inf detected in logits[{i}], replacing...")
                logits[i] = torch.zeros_like(l)


        value = td["_value_"]
        state = td["state"]
        if state is not None:
            split_size = self.core_num_layers
            state = (state[:split_size], state[split_size:])

        phi = self.get_shared_embedding(x)
        # Detach to prevent inverse dynamics gradients from flowing back
        phi_detached = phi.detach()

        # Make sure e3b is on the same device before passing to _e3b_update
        if e3b is not None:
            e3b = e3b.to(device)

        e3b, intrinsic_reward = self._e3b_update(phi_detached, e3b)

        action, logprob, entropy = sample_logits(logits, action, False)
        return action, logprob, entropy, value, state, e3b, intrinsic_reward

    def forward(self, x, state=None, action=None, e3b=None):
        return self.get_action_and_value(x, state, action, e3b)

    def compute_inverse_dynamics_loss(self, current_obs, next_obs, action_type_targets, action_param_targets):
        # Compute embeddings using the shared encoder and detach.
        phi_current = self.get_shared_embedding(current_obs).detach()
        phi_next = self.get_shared_embedding(next_obs).detach()
        logits_type, logits_param = self.inverse_dynamics_model(phi_current, phi_next)
        loss_type = F.cross_entropy(logits_type, action_type_targets.long())
        loss_param = F.cross_entropy(logits_param, action_param_targets.long())
        return loss_type + loss_param

    def _e3b_update(self, phi, e3b):
        intrinsic_reward = None
        if e3b is not None:
            u = phi.unsqueeze(1) @ e3b
            intrinsic_reward = u @ phi.unsqueeze(2)
            e3b = 0.99*e3b - (u.mT @ u) / (1 + intrinsic_reward)
            intrinsic_reward = intrinsic_reward.squeeze()
        return e3b, intrinsic_reward

    def l2_reg_loss(self) -> torch.Tensor:
        l2_reg_loss = 0
        for component in self.components.values():
            l2_reg_loss += component.l2_reg_loss() or 0
        return torch.tensor(l2_reg_loss)

    def l2_init_loss(self) -> torch.Tensor:
        l2_init_loss = 0
        for component in self.components.values():
            l2_init_loss += component.l2_init_loss() or 0
        return torch.tensor(l2_init_loss)

    def update_l2_init_weight_copy(self):
        for component in self.components.values():
            component.update_l2_init_weight_copy()

    def clip_weights(self):
        if self.clip_range > 0:
            for component in self.components.values():
                component.clip_weights()

    def compute_effective_rank(self, delta: float = 0.01) -> List[dict]:
        effective_ranks = []
        for component in self.components.values():
            rank = component.effective_rank(delta)
            if rank is not None:
                effective_ranks.append(rank)
        return effective_ranks
