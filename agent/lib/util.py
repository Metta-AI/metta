import hashlib
from torch import nn
import numpy as np
import torch
from tensordict import TensorDict


class MettaComponent(nn.Module):
    def __init__(self, component_cfg, components):
        super().__init__()
        self.components = components
        # check if self.components updates as the we append to the list (pointer) or is static from init
        self.component_cfg = component_cfg
        self.name = component_cfg.name
        # default to the input source being the last component
        self.input_source = component_cfg.input if component_cfg.input else self.components[-1].name
        self.layer_type = component_cfg.layer_type if component_cfg.layer_type else 'Linear'
        self.nonlinearity = component_cfg.nonlinearity if component_cfg.nonlinearity else 'ReLU'

    def get_input_source_size(self):
        self.input_size = self.input_source.get_out_size()
        # default to output size of the input
        self.output_size = self.component_cfg.output if self.component_cfg.output else self.input_size

    def initialize_layer(self):
        if self.layer_type == 'Linear':
            self.layer = nn.Linear(self.input, self.output)
        elif self.layer_type == 'Dropout':
            if 'dropout_prob' not in self.component_cfg:
                self.component_cfg.dropout_prob = 0.2
            self.layer = nn.Dropout(self.component_cfg.dropout_prob)
        elif self.layer_type == 'Conv2d':
            if 'kernel_size' not in self.component_cfg:
                self.component_cfg.kernel_size = 3
            if 'stride' not in self.component_cfg:
                self.component_cfg.stride = 1       
            if 'padding' not in self.component_cfg:
                self.component_cfg.padding = 1
            self.layer = nn.Conv2d(self.input, self.output, kernel_size=self.component_cfg.kernel_size, stride=self.component_cfg.stride, padding=self.component_cfg.padding)
        else:
            raise ValueError(f"Invalid layer type: {self.layer_type}")
        
        if self.nonlinearity:
            self.layer = nn.Sequential(self.layer, getattr(nn, self.nonlinearity)())

        # initialize weights
        # change below to have exposed functions 
        # pass exposed functions to MettaNet, MettaAgent, and PAW
        if 'clip_scale' in self.component_cfg:
            self.layer = nn.utils.clip_grad_norm_(self.layer, self.component_cfg.clip_scale)
        if 'normalize_weights' in self.component_cfg:
            self.layer = nn.utils.normalize_weights(self.layer)
        if 'clip_weights' in self.component_cfg:
            self.layer = nn.utils.clip_weights(self.layer)
        if 'l2_norm_scale' in self.component_cfg:
            self.layer = nn.utils.l2_norm_scale(self.layer, self.component_cfg.l2_norm_scale)
        if 'effective_rank' in self.component_cfg:
            self.layer = nn.utils.effective_rank(self.layer, self.component_cfg.effective_rank)

    def forward(self, x):
        return self.layer(x)
    
    def get_out_size(self):
        return self.output_size
    
class MettaNet(nn.Module):
    def __init__(self, components, output_name):
        super().__init__()
        self.components = components
        self.output_name = output_name
        
        for comp in self.components:
            if comp.name == self.output_name:
                current = comp
                break
        
        chain = []
        while True:
            chain.append(current)
            if current.input_source is None:
                break
            
            higher_comp = None
            for c in self.components:
                if c.name == current.input_source:
                    higher_comp = c
                    break
            
            current = higher_comp
        chain.reverse()
        self._forward_path = nn.ModuleList(chain)

    def forward(self, td: TensorDict):
        for comp in self._forward_path:
            if comp.name == '_obs_':
                td = comp(td["obs"])
            elif comp.name == '_recurrent_':
                td = comp(td["core_output"])
            else:
                td = comp(td)
        return td

class _MettaHelperComponent(nn.Module):
    def __init__(self, name, output_name, output_size):
        super().__init__()
        self.name = name
        self.output_name = output_name
        self.output_size = output_size
        self.input_source = None
    
    def get_input_source_size(self):
        pass
    
    def initialize_layer(self):
        pass
    #should this have nn.Identity instead?
    def forward(self, td: TensorDict):
        x = td[self.input_name]
        return td
    
    def get_out_size(self):
        return self.output_size

# do we still need this?
def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    layer_norm=False,
    use_skip=False,
):
    """Create a stack of fully connected layers with nonlinearity"""
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))

        if i < len(sizes) - 1:
            layers.append(nonlinearity)

        if layer_norm and i < len(sizes) - 1:
            layers.append(nn.LayerNorm(sizes[i]))

    if use_skip:
        return SkipConnectionStack(layers)
    else:
        return nn.Sequential(*layers)

class SkipConnectionStack(nn.Module):
    def __init__(self, layers):
        super(SkipConnectionStack, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if len(skip_connections) > 1:
                    x = x + skip_connections[-1]
                skip_connections.append(x)
            x = layer(x)
        return x


def stable_hash(s, mod=10000):
    """Generate a stable hash for a string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

def embed_strings(string_list, embedding_dim):
    return torch.tensor([
        embed_string(s, embedding_dim)
        for s in string_list
    ], dtype=torch.float32)

def embed_string(s, embedding_dim=128):
    # Hash the string using SHA-256, which produces a 32-byte hash
    hash_object = hashlib.sha256(s.encode())
    hash_digest = hash_object.digest()

    # Convert hash bytes to a numpy array of floats
    # This example simply takes the first 'embedding_dim' bytes and scales them
    byte_array = np.frombuffer(hash_digest[:embedding_dim], dtype=np.uint8)
    embedding = byte_array / 255.0  # Normalize to range [0, 1]

    # Ensure the embedding is the right size
    if len(embedding) < embedding_dim:
        # If the hash is smaller than the needed embedding size, pad with zeros
        embedding = np.pad(embedding, (0, embedding_dim - len(embedding)), 'constant')

    return embedding

def name_to_activation(activation: str):
    match activation:
        case 'relu':
            return nn.ReLU()
        case 'leaky_relu':
            return nn.LeakyReLU()
        case 'tanh':
            return nn.Tanh()
        case _:
            raise ValueError(f"Unknown activation: {activation}")
