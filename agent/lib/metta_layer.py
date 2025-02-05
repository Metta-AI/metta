from __future__ import annotations

import omegaconf
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
import torch

class MettaLayer(nn.Module):
    def __init__(self, MettaAgent, **cfg):
        cfg = OmegaConf.create(cfg)
        super().__init__()
        self.MettaAgent = MettaAgent
        self.cfg = cfg
        self.name = cfg.name
        self.input_source = cfg.input_source
        self.output_size = cfg.get('output_size', None)
        self.layer_type = cfg.get('layer_type', 'Linear')
        self.nonlinearity = cfg.get('nonlinearity', 'ReLU')
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

        # --- initialize your layer ---
        # can this be simpler?
        if self.layer_type == 'Linear':
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size)
        elif self.layer_type == 'Conv2d':
            # input size is the number of objects
            # output size is the number of channels
            self.layer = getattr(nn, self.layer_type)(self.input_size, self.output_size, self.cfg.kernel_size, self.cfg.stride)
        elif self.layer_type == 'Dropout':
            self.layer = getattr(nn, self.layer_type)(self.cfg.dropout_prob)
        elif self.layer_type == 'BatchNorm2d':
            self.layer = getattr(nn, self.layer_type)(self.input_size)
        elif self.layer_type == 'Identity':
            self.layer = nn.Identity()
        elif self.layer_type == 'Flatten':
            self.layer = nn.Flatten()
        else:
            raise ValueError(f"Layer type {self.layer_type} not supported")
        # add resnet, etc.

        # need to offer the ability to append a nonlinear layer for these layers other than Linear and Conv2d

        #add initialization

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
        attributes = ['clip_scale', 'l2_norm_scale', 'l2_init_scale', 'effective_rank', 'initialization', 'nonlinearity']
        for attr in attributes:
            if attr in self.cfg:
                setattr(self, attr, self.cfg[attr])
            else:
                setattr(self, attr, None)

        if self.nonlinearity:
            self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())

    def clip_weights(self):
        # need to feed global clip value
        if self.clip_scale:
            self.layer.weight.data = self.layer.weight.data.clamp(-self.clip_scale, self.clip_scale)

    def l2_norm_weights(self):
        if self.l2_norm_scale:
            self.layer.weight.data = self.layer.weight.data / self.l2_norm_scale

    def l2_init_weights(self):
        if self.l2_init_scale:
            self.layer.weight.data = self.layer.weight.data / self.l2_init_scale

    def effective_rank(self):
        if self.effective_rank:
            self.layer.weight.data = self.layer.weight.data / self.effective_rank

    def initialization(self):
        if self.initialization:
            self.layer.weight.data = self.layer.weight.data / self.initialization

class MettaLayerBase(nn.Module):
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

        First, recursively compute the input to the layer above this layer.
        Skip this if you are a top layer.
        x = self.MettaAgent.components[self.input_source].forward(td)

        Compute this layer's output.
        x = self.layer(x)

        Write your layer's name on your output so the next layer can find it.
        td[self.name] = x

        Pass the full td back.
        return td
        '''
        raise NotImplementedError(f"The method forward() is not implemented yet for object {self.__class__.__name__}.")