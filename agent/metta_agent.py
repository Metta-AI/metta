from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from pdb import set_trace as T
from torch.distributions.utils import logits_to_probs

from sample_factory.utils.typing import ActionSpace, ObsSpace
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


class MettaAgent(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
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

        e3b, intrinsic_reward = self._e3b_update(td["_core_"].detach(), e3b)

        action, logprob, entropy, normalized_logits = self.sample_logits(logits, action)
        return action, logprob, entropy, value, state, e3b, intrinsic_reward, normalized_logits
    
    def forward(self, x, state=None, action=None, e3b=None):
        return self.get_action_and_value(x, state, action, e3b)
    
    def _e3b_update(self, phi, e3b):
        intrinsic_reward = None
        if e3b is not None:
            u = phi.unsqueeze(1) @ e3b
            intrinsic_reward = u @ phi.unsqueeze(2)
            e3b = 0.99*e3b - (u.mT @ u) / (1 + intrinsic_reward)
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
        print(f"Effective ranks: {effective_ranks}")
        return effective_ranks

# ------------------From cleanrl.py--------------------------------

    def log_prob(self, logits, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self, logits):
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * logits_to_probs(logits)
        return -p_log_p.sum(-1)

    def sample_logits(self, logits: Union[torch.Tensor, List[torch.Tensor]],
            action=None):
        is_discrete = isinstance(logits, torch.Tensor)
        if isinstance(logits, torch.Tensor):
            normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
            logits = [logits]
        else:
            normalized_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits]

        if action is None:
            action = torch.stack([torch.multinomial(logits_to_probs(l), 1).squeeze() for l in logits])
        else:
            batch = logits[0].shape[0]
            action = action.view(batch, -1).T

        assert len(logits) == len(action)

        logprob = torch.stack([self.log_prob(l, a) for l, a in zip(normalized_logits, action)]).T.sum(1)
        logits_entropy = torch.stack([self.entropy(l) for l in normalized_logits]).T.sum(1)

        if is_discrete:
            return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), normalized_logits.squeeze(0)

        return action.T, logprob, logits_entropy, normalized_logits
