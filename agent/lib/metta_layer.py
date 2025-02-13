from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
import torch
import numpy as np
import weakref
class LayerBase(nn.Module):
    def __init__(self, metta_agent, **cfg):
        cfg = OmegaConf.create(cfg)
        super().__init__()
        # self.metta_agent = weakref.ref(metta_agent)
        object.__setattr__(self, 'metta_agent', metta_agent)
        self.cfg = cfg
        self.name = cfg.name
        self.input_source = cfg.get('input_source', None)
        self.output_size = cfg.get('output_size', None)

    def setup_layer(self):
        # delete this
        print(f"input source: {self.input_source}")
        if self.input_source is None:
            if self.output_size is None:
                raise ValueError(f"Neither input source nor output size is set for layer {self.name}")
        else:
            self.metta_agent.components[self.input_source].setup_layer()
            self.input_size = self.metta_agent.components[self.input_source].output_size

        if self.output_size is None:
            self.output_size = self.input_size

        self._initialize_layer()

    def _initialize_layer(self):
        pass

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source is not None:
            self.metta_agent.components[self.input_source].forward(td)

        td[self.name] = self.layer(td[self.input_source])

        return td
        
    def clip_weights(self):
        pass
    def l2_reg_loss(self):
        pass
    def l2_init_loss(self):
        pass
    def update_l2_init_weight_copy(self):
        pass
    def effective_rank(self, delta: float = 0.01) -> dict:
        pass
    
class ParameterizedLayer(LayerBase):
    def __init__(self, metta_agent, **cfg): 
        cfg = OmegaConf.create(cfg)
        # this should be only for the layer that calls this
            # for key, value in cfg.items():
            #     setattr(self, key, value)
        self.clip_scale = self.cfg.get('clip_scale', None)
        self.effective_rank = self.cfg.get('effective_rank', None)
        self.l2_norm_scale = self.cfg.get('l2_norm_scale', None)
        self.l2_init_scale = self.cfg.get('l2_init_scale', None)
        self.nonlinearity = self.cfg.get('nonlinearity', 'ReLU')
        self.initialization = self.cfg.get('initialization', 'Orthogonal')
        self.global_clip_range = metta_agent.clip_range
        super().__init__(metta_agent, **cfg)

    def _parameter_layer_helper(self):
        self.weight_layer = self.layer
        self.largest_weight = self._initialize_weights()

        if self.clip_scale != 0 and self.clip_scale is not None:
            self.clip_value = self.global_clip_range * self.largest_weight * self.clip_scale
        else:
            self.clip_value = self.global_clip_range

        if self.l2_init_scale != 0:
            self.initial_weights = self.layer.weight.data.clone()
        else:
            self.initial_weights = None

        if self.nonlinearity is not None:
            self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())
            self.weight_layer = self.layer[0]

    def _initialize_weights(self):
        fan_in = self.input_size
        fan_out = self.output_size

        if self.initialization.lower() == 'orthogonal':
            if self.nonlinearity == 'Tanh':
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(self.weight_layer.weight, gain=gain)
            largest_weight = self.weight_layer.weight.max().item()
        elif self.initialization.lower() == 'xavier':
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(self.weight_layer.weight)
        elif self.initialization.lower() == 'normal':
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(self.weight_layer.weight, mean=0, std=largest_weight)
        elif self.initialization.lower() == 'max_0_01':
            #set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(self.weight_layer.weight, a=-largest_weight, b=largest_weight)
        else:
            raise ValueError(f"Invalid initialization method: {self.initialization}")

        if hasattr(self.weight_layer, "bias") and isinstance(self.weight_layer.bias, torch.nn.parameter.Parameter):
            self.weight_layer.bias.data.fill_(0)

        return largest_weight

    def clip_weights(self):
        if self.clip_scale != 0:
            with torch.no_grad():
                self.weight_layer.weight.data = self.weight_layer.weight.data.clamp(-self.clip_value, self.clip_value)

    def l2_reg_loss(self) -> torch.Tensor:
        '''
        Also known as Weight Decay Loss or L2 Ridge Regularization
        '''
        l2_reg_loss = torch.tensor(0.0, device=self.weight_layer.weight.data.device)
        if self.l2_norm_scale != 0 and self.l2_norm_scale is not None:
            l2_reg_loss = (torch.sum(self.weight_layer.weight.data ** 2))*self.l2_norm_scale
        return l2_reg_loss

    def l2_init_loss(self) -> torch.Tensor:
        '''
        Also known as Delta Regularization Loss
        '''
        l2_init_loss = torch.tensor(0.0, device=self.weight_layer.weight.data.device)
        if self.l2_init_scale != 0 and self.l2_init_scale is not None:
            l2_init_loss = torch.sum((self.weight_layer.weight.data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss
    
    def update_l2_init_weight_copy(self, alpha: float = 0.9):
        '''
        Potentially useful to prevent catastrophic forgetting.
        Update the initial weights copy with a weighted average of the previous and current weights.
        '''
        if self.initial_weights is not None:
            self.initial_weights = (self.initial_weights * alpha + self.weight_layer.weight.data * (1 - alpha)).clone()
    
    def effective_rank(self, delta: float = 0.01) -> dict:
        """
        Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
        """
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(self.weight_layer.weight.data.detach())
        
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
    def __init__(self, metta_agent, **cfg):
        super().__init__()
        self.metta_agent = metta_agent
        self.cfg = cfg
        #required attributes
        self.name = None
        self.input_source = None
        self.output_size = None

    def setup_layer(self):
        '''
        Recursively set the input size for the component above your layer.
        This is necessary unless you are a top layer, in which case, you can skip this.
        self.metta_agent.components[self.input_source].setup_layer()
        
        Set your input size to be the output size of the layer above you or otherwise ensure that this is the case.
        self.input_size = self.metta_agent.components[self.input_source].output_size

        With your own input and output sizes set, initialize your layer, if necessary.
        self.layer = ...

        '''
        raise NotImplementedError(f"The method setup_layer() is not implemented yet for object {self.__class__.__name__}.")

    def forward(self, td: TensorDict):
        '''
        First, ensure we're not recomputing in case your layer is already computed.
        if self.name in td:
            return td[self.name]

        First, recursively compute the input to the layers above this layer.
        Skip this if you are a top layer.
        if isinstance(self.input_source, list):
            for src in self.input_source:
                self.metta_agent.components[src].forward(td) 
        else:
            self.metta_agent.components[self.input_source].forward(td)

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
