from __future__ import annotations

import omegaconf
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
import torch
import numpy as np
class MettaLayer(nn.Module):
    def __init__(self, MettaAgent, **cfg):
        cfg = OmegaConf.create(cfg)
        super().__init__()
        self.MettaAgent = MettaAgent
        self.cfg = cfg
        self.name = cfg.name
        self.input_source = cfg.input_source
        self.output_size = cfg.get('output_size', None)
        self.clip_multiplier = cfg.get('clip_multiplier', None)
        # can change the above to default to None and then handle it in the parameter_layer_helper

    def set_input_size_and_initialize_layer(self):
        if self.input_source == '_obs_':
            self.input_size = self.MettaAgent.num_objects
        elif self.input_source == '_core_':
            self.input_size = self.output_size
        else:
            if isinstance(self.input_source, omegaconf.listconfig.ListConfig):
                self.input_source = list(self.input_source)
                for src in self.input_source:
                    self.MettaAgent.components[src].set_input_size_and_initialize_layer()

                self.input_size = sum(self.MettaAgent.components[src].output_size for src in self.input_source)
            else:
                self.MettaAgent.components[self.input_source].set_input_size_and_initialize_layer()
                self.input_size = self.MettaAgent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

        self.instantiate_layer_from_cfg()

        # Layer initialization mapping
        # layer_params = {
        #     'Linear': (self.input_size, self.output_size),
        #     'Conv2d': (self.input_size, self.output_size, self.cfg.kernel_size, self.cfg.stride),
        #     'Dropout': (self.cfg.dropout_prob,),
        #     'BatchNorm2d': (self.input_size,),
        #     'Identity': (),
        #     'Flatten': ()
        # }

        # self.layer_type = self.cfg.get('layer_type', 'Linear')
        
        # if self.layer_type not in layer_params:
        #     raise ValueError(f"Layer type {self.layer_type} not supported")

        # self.layer = getattr(nn, self.layer_type)(*layer_params[self.layer_type])
        
        # if self.layer_type in ['Linear', 'Conv2d']:
        #     self.parameter_layer_helper()
        #     self.initialize_layer()

    def instantiate_layer_from_cfg(self):
        '''
        nn_params_dict key names should be the same as the argument names of the torch nn class.
        We add nonlinear layers to Linear and Conv classes. Specify nonlinearity = None in the cfg to change this.
        You can also add a nonlinear layer as a standalone by supplying it as the layer_type in the cfg.
        '''
        nn_params_dict = self.cfg.get('nn_params', {})

        # manually handle the few cases that require special signature mappings of sizes from other layers
        layer_signature_map = {
            'Linear': (self.input_size, self.output_size),
            'Conv': (self.input_size, self.output_size),
            'BatchNorm': (self.input_size,),
        }

        base_layer_type = next((key for key in layer_signature_map if self.layer_type.startswith(key)), None)

        if base_layer_type:
            base_params = layer_signature_map[base_layer_type]
            self.layer = getattr(nn, self.layer_type)(*base_params, **nn_params_dict)
        else:
            self.layer = getattr(nn, self.layer_type)(**nn_params_dict)

        if any(substring in self.layer_type for substring in ['Linear', 'Conv']):
            self.parameter_layer_helper()
            self.initialize_layer()

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source == '_obs_':
            td[self.name] = td["obs"][self.MettaAgent.obs_key]
        elif self.input_source == '_core_':
            td[self.name] = td["core_output"]
        else:
# need to think about cat vs add vs subtract
            if isinstance(self.input_source, list):
                for src in self.input_source:
                   self.MettaAgent.components[src].forward(td) 
            else:
                self.MettaAgent.components[self.input_source].forward(td)

            if isinstance(self.input_source, list):
                inputs = [td[src] for src in self.input_source]
                x = torch.cat(inputs, dim=-1)
                td[self.name] = self.layer(x)
            else:
                td[self.name] = self.layer(td[self.input_source])

        return td
    
    def parameter_layer_helper(self):
        attributes = ['clip_scale', 'l2_norm_scale', 'l2_init_scale', 'effective_rank', 'initialization']
        for attr in attributes:
            if attr in self.cfg:
                setattr(self, attr, self.cfg[attr])
            else:
                setattr(self, attr, None)

        largest_weight = self.initialize_layer()

        if self.clip_scale:
            self.clip_value = self.MettaAgent.clip_multiplier * self.largest_weight * self.clip_scale

        if self.l2_init_scale:
            self.initial_weights = self.layer.weight.data.clone()

# is this redundant with attributes above?
        self.nonlinearity = self.cfg.get('nonlinearity', 'ReLU')
        self.weights_data = self.layer.weight.data
        if self.nonlinearity is not None:
            self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())
            self.weights_data = self.layer[0].weight.data


    def initialize_layer(self):
        '''
        Assumed that this is run before appending a nonlinear layer.
        '''
        fan_in, fan_out = self.layer.weight.shape

        if self.initialization is None or self.initialization == 'Orthogonal':
            if self.nonlinearity == 'Tanh':
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(self.layer.weight, gain=gain)
            largest_weight = self.layer.weight.max().item()
        elif self.initialization == 'Xavier':
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(self.layer.weight)
        elif self.initialization == 'Normal':
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(self.layer.weight, mean=0, std=largest_weight)
        elif self.initialization == 'Max_0_01':
            #set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(self.layer.weight, a=-largest_weight, b=largest_weight)

        if hasattr(self.layer, "bias") and isinstance(self.layer.bias, torch.nn.parameter.Parameter):
            self.layer.bias.data.fill_(0)

        return largest_weight

    def clip_weights(self):
        if self.clip_scale:
            with torch.no_grad():
                self.weights_data = self.weights_data.clamp(-self.clip_value, self.clip_value)

    def get_l2_reg_loss(self) -> torch.Tensor:
        '''
        Also known as Weight Decay Loss or L2 Ridge Regularization
        '''
        l2_reg_loss = torch.tensor(0.0, device=self.weights_data.device)
        if self.l2_norm_scale:
            l2_reg_loss = (torch.sum(self.weights_data ** 2))*self.l2_norm_scale
        return l2_reg_loss

    def get_l2_init_loss(self) -> torch.Tensor:
        '''
        Also known as Delta Regularization Loss
        '''
        l2_init_loss = torch.tensor(0.0, device=self.weights_data.device)
        if self.l2_init_scale:
            l2_init_loss = torch.sum((self.weights_data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss
    
    def update_l2_init_weight_copy(self, alpha: float = 0.9):
        '''
        Potentially useful to prevent catastrophic forgetting.
        Update the initial weights copy with a weighted average of the previous and current weights.
        '''
        self.initial_weights = (self.initial_weights * alpha + self.weights_data * (1 - alpha)).clone()
    
    def get_effective_rank(self, delta: float = 0.01):
        """
        Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
        """
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(self.weights_data.detach())
        
        # Calculate the cumulative sum of singular values
        total_sum = S.sum()
        cumulative_sum = torch.cumsum(S, dim=0)
        
        # Find the smallest k that satisfies the effective rank condition
        threshold = (1 - delta) * total_sum
        effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
        
        return {'name': self.name, 'effective_rank': effective_rank}



class MettaLayerBase(nn.Module):
    '''
    This is the base class and instructions for custom layers in the stead of MettaLayers.
    '''
    def __init__(self, MettaAgent, **cfg):
        super().__init__()
        self.MettaAgent = MettaAgent
        self.cfg = cfg
        #required attributes
        self.name = None
        self.input_source = None
        self.output_size = None

    def set_input_size_and_initialize_layer(self):
        '''
        Recursively set the input size for the component above your layer.
        This is necessary unless you are a top layer, in which case, you can skip this.
        self.MettaAgent.components[self.input_source].set_input_source_size()
        
        Set your input size to be the output size of the layer above you or otherwise ensure that this is the case.
        self.input_size = self.MettaAgent.components[self.input_source].output_size

        With your own input and output sizes set, initialize your layer, if necessary.
        self.layer = ...

        '''
        raise NotImplementedError(f"The method set_input_source_size() is not implemented yet for object {self.__class__.__name__}.")

    def forward(self, td: TensorDict):
        '''
        First, ensure we're not recomputing in case your layer is already computed.
        if self.name in td:
            return td[self.name]

        First, recursively compute the input to the layers above this layer.
        Skip this if you are a top layer.
        if isinstance(self.input_source, list):
            for src in self.input_source:
                self.MettaAgent.components[src].forward(td) 
        else:
            self.MettaAgent.components[self.input_source].forward(td)

        Compute this layer's output (assuming you have a .layer attribute).
        Write your layer's name on your output so the next layer can find it.
        if isinstance(self.input_source, list):
            inputs = [td[src] for src in self.input_source]
            x = torch.cat(inputs, dim=-1)
            td[self.name] = self.layer(x)
        else:
            td[self.name] = self.layer(td[self.input_source])

        Pass the full td back.
        return td
        '''
        raise NotImplementedError(f"The method forward() is not implemented yet for object {self.__class__.__name__}.")