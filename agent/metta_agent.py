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
        # obs_width=11,
        # obs_height=11, # TODO: remove hardcoded values
        # # obs_width=cfg.env.game.obs_width,
        # # obs_height=cfg.env.game.obs_height,
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
        self.convert_to_single_discrete = cfg.get('convert_to_single_discrete', True)

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

        # delete if logic after testing
        if self.convert_to_single_discrete:
            component = self.components['_action_']
            self._setup_components(component)
        else:
            component = self.components['_action_type_']
            self._setup_components(component)
            component = self.components['_action_param_']
            self._setup_components(component)

        for name, component in self.components.items():
            if not getattr(component, 'ready', False):
                raise RuntimeError(f"Component {name} in MettaAgent was never setup. It might not be accessible by other components.")

        self.components = self.components.to(device)
        self.device = device

        self._total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in MettaAgent: {self._total_params:,}. Setup complete.")
        
    def _setup_components(self, component):
        if component._input_source is not None:
            if isinstance(component._input_source, omegaconf.listconfig.ListConfig):
                component._input_source = list(component._input_source)

            if isinstance(component._input_source, list):
                for input_source in component._input_source:
                    self._setup_components(self.components[input_source])
            else:
                self._setup_components(self.components[component._input_source])

        if component._input_source is not None:
            if isinstance(component._input_source, list):
                input_source_components = {}
                for name in component._input_source:
                    input_source_components[name] = self.components[name]
                component.setup(input_source_components)
            else:
                component.setup(self.components[component._input_source])
        else:
            component.setup()

        # delete after testing
        print((
            f"Component: {component._name}, in name: {component._input_source}, "
            f"in_size: {getattr(component, '_in_tensor_shape', 'None')}, out_size: {getattr(component, '_out_tensor_shape', 'None')}"
        ))

    def activate_actions(self, action_names, action_max_params):
        '''Run this at the beginning of training.'''
        self.actions_max_params = action_max_params
        self.active_actions = list(zip(action_names, action_max_params))
        
        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.tensor([0] + [sum(action_max_params[:i+1]) for i in range(len(action_max_params))], 
                                                 device=self.device)
        
        # delete if logic after testing
        if self.convert_to_single_discrete:
            # convert the actions_dict into a list of strings
            string_list = []
            for action_name, max_arg_count in self.active_actions:
                for i in range(max_arg_count + 1):
                    string_list.append(f"{action_name}_{i}")
            self.components['_action_embeds_'].activate_actions(string_list)
        else:
            self.components['_action_type_embeds_'].activate_actions(action_names)
            param_list = []
            for i in range(max(action_max_params) - 1):
                param_list.append(str(i))
            self.components['_action_param_embeds_'].activate_actions(param_list)

        # Create action_index tensor instead of list
        action_index = []
        action_type_number = 0
        for max_param in action_max_params:
            for j in range(max_param+1):
                action_index.append([action_type_number, j])
            action_type_number += 1
            
        self.action_index_tensor = torch.tensor(action_index, device=self.device)
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

    def _convert_action_to_logit_index(self, action, logits):
        """Convert action pairs to logit indices using vectorized operations"""
        orig_shape = action.shape
        action = action.reshape(-1, 2)
        
        # Extract action components without list comprehension
        action_type_numbers = action[:, 0].long()
        action_params = action[:, 1].long()
        
        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        
        # Vectorized addition
        action_logit_index = action_type_numbers + cumulative_sum + action_params
        
        return action_logit_index.reshape(*orig_shape[:2], 1)

    def _convert_logit_index_to_action(self, action_logit_index, td):
        """Convert logit indices back to action pairs using tensor indexing"""
        if td.get("_TT_", 0) == 1:  # means we are in rollout, not training
            # Use direct tensor indexing on precomputed action_index_tensor
            return self.action_index_tensor[action_logit_index.reshape(-1)]

    def get_action_and_value(self, x, state=None, action=None, e3b=None):
        td = TensorDict({"x": x})

        td["state"] = None
        if state is not None:
            state = torch.cat(state, dim=0)
            td["state"] = state.to(x.device)

        self.components["_value_"](td)
        value = td["_value_"]
        state = td["state"]

        # delete if logic after testing
        if self.convert_to_single_discrete:
            self.components["_action_"](td)
            logits = td["_action_"]
        else:  
            self.components["_action_type_"](td)
            self.components["_action_param_"](td)
            logits = [td["_action_type_"], td["_action_param_"]]

        if state is not None:
            split_size = self.core_num_layers
            state = (state[:split_size], state[split_size:])

        e3b, intrinsic_reward = self._e3b_update(td["_core_"].detach(), e3b)

        # delete if logic after testing
        if self.convert_to_single_discrete:
            action_logit_index_provided = self._convert_action_to_logit_index(action, logits) if action is not None else None
            sampled_action_logit_index, logprob, entropy, normalized_logits = sample_logits(logits, action_logit_index_provided)

            # Only convert the sampled index back to action pair format if we didn't have an action provided (i.e., during evaluation/rollout)
            if action is None:
                action_to_return = self._convert_logit_index_to_action(sampled_action_logit_index, td)
            else:
                # During training, we don't need the converted action, just return the sampled index (or original action if needed elsewhere, though likely not)
                # Keep sampled_action_logit_index as it aligns with logprob/entropy for loss calculation
                action_to_return = sampled_action_logit_index
        else:
            # Assuming similar logic: sample_logits returns the action/index needed for loss
            action_logit_index, logprob, entropy, normalized_logits = sample_logits(logits, action)
            action_to_return = action_logit_index

        return action_to_return, logprob, entropy, value, state, e3b, intrinsic_reward, normalized_logits

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
