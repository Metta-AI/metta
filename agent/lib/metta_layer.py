from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
import torch
import numpy as np

class LayerBase(nn.Module):
    '''The base class for layers that make up the metta agent. All layers are
    required to have a name and an input source, although the input source can
    be None (or null in your YAML). Output size is optional depending on your
    layer.

    All layers must have a setup_layer method, and it must recursively call
    setup_layer on its own input_source. setup_layer assigns the output_size 
    if it is not already set. Once it has been called on all layers, layers 
    can initialize their parameters via _initialize_layer, if necessary.

    The forward method should only take a tensordict as input and return only
    the tensordict. The tensor dict is constructed anew each time the
    metta_agent forward pass is run. The layer's forward should read from the
    value at the key name of its input_source. After performing its computation
    via self.layer or otherwise, it should store its output in the tensor dict
    at the key with its own name.

    Before doing this, it should first check if the tensordict already has a
    key with its own name, indicating that its computation has already been
    performed (due to some other run up the DAG) and return. After this check,
    it should check if its input source is not None and recursively call the
    forward method of the layer above it.'''
    def __init__(self, name, input_source=None, output_size=None, nn_params={}, **cfg):
        super().__init__()
        self.name = name
        self.input_source = input_source
        self.output_size = output_size
        self.nn_params = nn_params
        self.net = None
        self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup(self, input_source_component=None):
        if self._ready:
            return

        self.input_source_component = input_source_component

        if self.input_source_component is None:
            self.input_size = None
            if self.output_size is None: # output size must be set for a top level component
                raise ValueError(f"Either input source or output size must be set for layer {self.name}")
        else:
            self.input_size = self.input_source_component.output_size

        if self.output_size is None:
            self.output_size = self.input_size

        self._initialize()
        self._ready = True

    def _initialize(self):
        self.net = self._make_net()

    def _make_net(self):
        pass

    def forward(self, td: TensorDict):
        if self.name in td:
            return td

        if self.input_source_component is not None:
            self.input_source_component.forward(td)

        self._forward(td)

        return td
    
    def _forward(self, td: TensorDict):
        td[self.name] = self.net(td[self.input_source])
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
    
class ParamLayer(LayerBase):
    '''This provides a few useful methods for components/nets that have parameters (weights).
    Superclasses should have input_size and output_size already set.'''
    def __init__(self, clip_scale=1, effective_rank=None, l2_norm_scale=None, l2_init_scale=None, nonlinearity='nn.ReLU', initialization='Orthogonal', clip_range=None, **cfg): 
        self.clip_scale = clip_scale
        self.effective_rank_bool = effective_rank
        self.l2_norm_scale = l2_norm_scale
        self.l2_init_scale = l2_init_scale
        self.nonlinearity = nonlinearity
        self.initialization = initialization
        self.global_clip_range = clip_range
        super().__init__(**cfg)

    def _initialize(self):
        self.weight_net = self._make_net()

        self._initialize_weights()

        if self.clip_scale is not None and self.clip_scale > 0:
            self.clip_value = self.global_clip_range * self.largest_weight * self.clip_scale
        else:
            self.clip_value = None

        if self.l2_init_scale != 0:
            self.initial_weights = self.weight_net.weight.data.clone()
        else:
            self.initial_weights = None

        if self.nonlinearity is not None:
            # expecting a string of the form 'nn.ReLU'
            try:
                _, class_name = self.nonlinearity.split('.')
                if class_name not in dir(nn):
                    raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")
                nonlinearity_class = getattr(nn, class_name)
                self.net = nn.Sequential(self.weight_net, nonlinearity_class())
                self.weight_net = self.net[0]
            except (AttributeError, KeyError, ValueError) as e:
                raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}") from e
        else:
            self.net = self.weight_net

    def _initialize_weights(self):
        fan_in = self.input_size
        fan_out = self.output_size

        if self.initialization.lower() == 'orthogonal':
            if self.nonlinearity == 'nn.Tanh':
                gain = np.sqrt(2)
            else:
                gain = 1
            nn.init.orthogonal_(self.weight_net.weight, gain=gain)
            largest_weight = self.weight_net.weight.max().item()
        elif self.initialization.lower() == 'xavier':
            largest_weight = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(self.weight_net.weight)
        elif self.initialization.lower() == 'normal':
            largest_weight = np.sqrt(2 / fan_in)
            nn.init.normal_(self.weight_net.weight, mean=0, std=largest_weight)
        elif self.initialization.lower() == 'max_0_01':
            #set to uniform with largest weight = 0.01
            largest_weight = 0.01
            nn.init.uniform_(self.weight_net.weight, a=-largest_weight, b=largest_weight)
        else:
            raise ValueError(f"Invalid initialization method: {self.initialization}")

        if hasattr(self.weight_net, "bias") and isinstance(self.weight_net.bias, torch.nn.parameter.Parameter):
            self.weight_net.bias.data.fill_(0)

        self.largest_weight = largest_weight

    def clip_weights(self):
        if self.clip_value is not None:
            with torch.no_grad():
                self.weight_net.weight.data = self.weight_net.weight.data.clamp(-self.clip_value, self.clip_value)

    def l2_reg_loss(self) -> torch.Tensor:
        '''Also known as Weight Decay Loss or L2 Ridge Regularization'''
        l2_reg_loss = torch.tensor(0.0, device=self.weight_net.weight.data.device)
        if self.l2_norm_scale != 0 and self.l2_norm_scale is not None:
            l2_reg_loss = (torch.sum(self.weight_net.weight.data ** 2))*self.l2_norm_scale
        return l2_reg_loss

    def l2_init_loss(self) -> torch.Tensor:
        '''Also known as Delta Regularization Loss'''
        l2_init_loss = torch.tensor(0.0, device=self.weight_net.weight.data.device)
        if self.l2_init_scale != 0 and self.l2_init_scale is not None:
            l2_init_loss = torch.sum((self.weight_net.weight.data - self.initial_weights) ** 2) * self.l2_init_scale
        return l2_init_loss
    
    def update_l2_init_weight_copy(self, alpha: float = 0.9):
        '''Potentially useful to prevent catastrophic forgetting. Update the 
        initial weights copy with a weighted average of the previous and 
        current weights.'''
        if self.initial_weights is not None:
            self.initial_weights = (self.initial_weights * alpha + self.weight_net.weight.data * (1 - alpha)).clone()
    
    def compute_effective_rank(self, delta: float = 0.01) -> dict:
        '''Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient 
        Deep Reinforcement Learning' by A. Kumar et al.'''
        if self.weight_net.weight.data.dim() != 2 or self.effective_rank_bool is None or self.effective_rank_bool == False:
            return None
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(self.weight_net.weight.data.detach())
        
        # Calculate the cumulative sum of singular values
        total_sum = S.sum()
        cumulative_sum = torch.cumsum(S, dim=0)
        
        # Find the smallest k that satisfies the effective rank condition
        threshold = (1 - delta) * total_sum
        effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
        
        return {'name': self.name, 'effective_rank': effective_rank}
