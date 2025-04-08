import logging
from typing import List, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import omegaconf
from pufferlib.environment import PufferEnv
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel


from agent.util.distribution_utils import sample_logits

logger = logging.getLogger("metta_agent")

def make_policy(env: PufferEnv, cfg: OmegaConf):
    obs_space = gym.spaces.Dict({
        "grid_obs": env.single_observation_space,
        "global_vars": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=[ 0 ],
            dtype=np.int32)
    })
    return hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.single_observation_space.shape,
        obs_space=obs_space,
        action_space=env.single_action_space,
        grid_features=env.grid_features,
        global_features=env.global_features,
        device=cfg.device,
        obs_width=11,
        obs_height=11, # TODO: remove hardcoded values
        # obs_width=cfg.env.game.obs_width,
        # obs_height=cfg.env.game.obs_height,
        _recursive_=False)

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
            'core_num_layers': self.core_num_layers,
        }
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
        component = self.components['_action_']
        self._setup_components(component)

        # component = self.components['_action_param_']
        # self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup. It might not be accessible by other components.")

        self.components = self.components.to(device)

        self._total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")
        
    def _setup_components(self, component):
        if component._input_source is not None:
            if isinstance(component._input_source, omegaconf.listconfig.ListConfig):
                component._input_source = list(component._input_source)

            if isinstance(component._input_source, list):
                for input_source in component._input_source:
                    self._setup_components(self.components[input_source])
            # elif isinstance(component._input_source, omegaconf.listconfig.ListConfig):
            #     component._input_source = list(component._input_source)
            #     for input_source in component._input_source:
            #         self._setup_components(self.components[input_source])
            else:
                self._setup_components(self.components[component._input_source])

        if component._input_source is not None:
            # path 1
            if isinstance(component._input_source, str):
                component.setup(self.components[component._input_source])
            elif isinstance(component._input_source, list):
                input_source_components = {}
                for name in component._input_source:
                    input_source_components[name] = self.components[name]
                component.setup(input_source_components)

            # path 2
            # if isinstance(component._input_source, list):
            #     input_source_components = {}
            #     for name in component._input_source:
            #         input_source_components[name] = self.components[name]
            #     component.setup(input_source_components)
            # else:
            #     component.setup(self.components[component._input_source])
        else:
            component.setup()

        # delete this after testing
        print((
            f"Component: {component._name}, in name: {component._input_source}, "
            f"in_size: {getattr(component, '_in_tensor_shape', 'None')}, out_size: {getattr(component, '_out_tensor_shape', 'None')}"
        ))

    def activate_actions(self, action_names, action_max_params):
        '''Run this at the beginning of training.'''
        self.actions_max_params = action_max_params
        self.active_actions = list(zip(action_names, action_max_params))
        self.components['_action_embeds_'].activate_actions(self.active_actions)

        self.action_index = [] # the list element number maps to the action type index
        action_type_number = 0
        for max_param in action_max_params:
            for j in range(max_param+1):
                self.action_index.append([action_type_number, j])
            action_type_number += 1
        print(f"Agent action index activated with: {self.active_actions}")

    @property
    def lstm(self):
        return self.components["_core_"]._net

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
        self.components["_action_"](td)

        logits = td["_action_"]
        value = td["_value_"]
        state = td["state"]

        # Convert state back to tuple to pass back to trainer
        if state is not None:
            split_size = self.core_num_layers
            state = (state[:split_size], state[split_size:])

        e3b, intrinsic_reward = self._e3b_update(td["_core_"].detach(), e3b)

        # convert action from a list of two elements to a single element
        # make this a function
        action_logit_index = None
        if action is not None:
            # Reshape action to [B*TT, 2] if it's [B, TT, 2]
            orig_shape = action.shape
            action = action.reshape(-1, 2) # is this necessary?
            
            # Convert each action pair to logit index
            action_type_numbers = torch.tensor([a[0] for a in action])
            action_params = torch.tensor([a[1].item() for a in action])
            cumulative_sum = torch.tensor([sum(self.actions_max_params[:num]) for num in action_type_numbers])
            action_logit_index = action_type_numbers + cumulative_sum + action_params
            
            # Reshape back to original batch dimensions and send to device
            action_logit_index = action_logit_index.reshape(*orig_shape[:2], 1).to(logits.device)

        action_logit_index, logprob, entropy, normalized_logits = sample_logits(logits, action_logit_index)
        
        # only need to do this on experience since training doesn't need action number
        # action = torch.tensor([self.action_index[idx.item()] for idx in action_logit_index])
        if td["_TT_"] == 1:
            action = torch.tensor([self.action_index[idx.item()] for idx in action_logit_index.reshape(-1)], 
                                  device=action_logit_index.device)
        # if td["_TT_"] > 1:
        #     # Reshape to [B, TT, 2]
        #     action = flat_actions.reshape(td["_batch_size_"], td["_TT_"], 2)
        # else:
        #     # Reshape to [B, 2]
        #     action = flat_actions.reshape(td["_batch_size_"], 2)

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
        return effective_ranks
