from __future__ import annotations

from typing import List

import hydra
from omegaconf import OmegaConf
from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sample_factory.model.core import ModelCoreRNN
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import Tensor
from sample_factory.algo.utils.action_distributions import sample_actions_log_probs

from tensordict import TensorDict
from torch import Tensor, nn
import torch
from agent.agent_interface import MettaAgentInterface
from agent.lib.util import make_nn_stack
# from agent.components import Composer, Layer
import omegaconf

class MettaAgent(nn.Module, MettaAgentInterface):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        grid_features: List[str],
        global_features: List[str],
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.observation_space = obs_space
        self.action_space = action_space

        self._encoder = hydra.utils.instantiate(
            cfg.observation_encoder,
            obs_space, grid_features, global_features)

        self._decoder = hydra.utils.instantiate(
            cfg.decoder,
            cfg.core.rnn_size)

        # self._critic_linear = make_nn_stack(
        #     self.decoder_out_size(),
        #     1,
        #     list(cfg.critic.hidden_sizes),
        #     nonlinearity=nn.ReLU()
        # )

        # self._critic = hydra.utils.instantiate(
        #     cfg.critic,
        #     input=self.decoder_out_size(),
        #     output=1,
        #     _recursive_=False)

        self._critic = Composer(
            self,
            cfg.critic)

        self.apply(self.initialize_weights)

    def decoder_out_size(self):
        return self._decoder.get_out_size()

    def encode_observations(self, td: TensorDict):
        td["encoded_obs"] = self._encoder(td["obs"])

    def decode_state(self, td: TensorDict):
        td["state"] = self._decoder(td["core_output"])
        td["values"] = self._critic_linear(td["state"]).squeeze()

    def decode_state_2(self, td: TensorDict):
        td["encoded_obs"] = self._encoder(td["obs"])
        #what's the diff between core_output and encoded_obs given that decoder is identity?
        td["state"] = self._decoder(td["core_output"], td["encoded_obs"])
        td["values"] = self._critic_linear(td["state"]).squeeze()


    def aux_loss(self, normalized_obs_dict, rnn_states):
        raise NotImplementedError()

    def initialize_weights(self, layer):
        gain = 1.0

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
            nn.init.orthogonal_(layer.weight.data, gain=gain)
        else:
            # LSTMs and GRUs initialize themselves
            # should we use orthogonal/xavier for LSTM cells as well?
            # I never noticed much difference between different initialization schemes, and here it seems safer to
            # go with default initialization,
            pass

class Layer(nn.Module):
    def __init__(self, layer_cfg):
        super().__init__()
        self.layer_type = layer_cfg.layer_type if 'layer_type' in layer_cfg else 'Linear'
        if self.layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'Softmax', 'ELU', 'GELU', 'SELU', 'Softplus', 'Softsign']:
            self.layer = getattr(nn, self.layer_type)()
            self.layer.name = layer_cfg.name
            return
        if self.layer_type == 'Dropout':
            self.layer = getattr(nn, self.layer_type)(layer_cfg.dropout_prob)
            self.layer.name = layer_cfg.name
            return
        self.input = layer_cfg.input
        self.output = layer_cfg.output
        self.layer = getattr(nn, self.layer_type)(layer_cfg.input, layer_cfg.output)
        self.name = layer_cfg.name
        self.input_source = layer_cfg.input_source

        
        # self.layer.initialize_weights()
        # self.layer.normalize_weights()
        # self.layer.clip_weights()
        # self.layer.get_losses()

    def forward(self, x):
        return self.layer(x)

    # we don't need this if it can just be a property, right?
    def get_out_size(self):
        return self.output
    

class Composer(nn.Module):
    def __init__(self, MettaAgent: MettaAgent, net_cfg):
    # def __init__(self, layers: ListConfig, input, output, MettaAgent: MettaAgent):
        super().__init__()
        self.MettaAgent = MettaAgent
        # self.input = self.get_size(net_cfg.input, "input")
        # self.output = self.get_size(net_cfg.output, "output")
        self.input_layers = list(net_cfg.layers)

        # self.input_layers[0].input = self.input
        # self.input_layers[-1].output = self.output
        # check if the other layers have input and output keys.
        # if not, set the input size to the output size of the previous layer and the output size to the same layer's input size
        self.layers = []
        for i in range(len(self.input_layers)):
            if 'input' not in self.input_layers[i]:
                self.input_layers[i].input = self.input_layers[i - 1].output
                self.input_layers[i].input_source = self.input_layers[i - 1].name
            else:
                self.input_layers[i].input_source = self.input_layers[i].input
                self.input_layers[i].input = self.get_size(self.input_layers[i].input, "input")

            if 'output' not in self.input_layers[i]:
                # default to output equals its input size
                self.input_layers[i].output = self.input_layers[i].input

            layer = Layer(self.input_layers[i])
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        print(self.layers)

    def get_size(self, value, type):
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            keys = value.split('.')
            if len(keys) > 1:
                # this should eventually walk the tree to get the layer within the net
                attr = getattr(self.MettaAgent, keys[-1], None)
                return attr.get_out_size()
            else:
                # find the self.layers index of the layer with the name value
                for i in range(len(self.layers)):
                    if self.layers[i].name == value:
                        return self.layers[i].get_out_size()
                raise ValueError(f"Layer with name {value} not found")


            # below, we want the out size of another net as our input size
            attr = getattr(self.MettaAgent, value, None)
            return attr.get_out_size()
        elif isinstance(value, omegaconf.listconfig.ListConfig):
            size = 0
            for layer in value:
                size += self.get_size(layer, type)
            return size
        else:
            raise ValueError(f"Invalid value type: {type(value)}")
        
    # need to figure out how to route the correct input in
    # maybe do this in MettaAgent?


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_out_size(self):
        return self.layers[-1].output_size
    
    def get_in_size(self):
        return self.layers[0].input_size