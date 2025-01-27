import hashlib
from torch import nn
import numpy as np
import torch

class WeightTransformer():
    def __init__(self, cfg):
        self._clip_scales = []
        self._l2_norm_scales = []
        self._effective_rank = []
        self._layers = []
        self.cfg = cfg
        self._global_clipping_coeff = cfg.clipping_coeff

    def add_layer(self, layer, key, layer_idx):
        layer_attributes = self._get_layer_attributes(self.cfg.get(key), layer_idx)

        s_bound = self._initialize(layer, layer_attributes['initialization'])

        self._clip_scales.append(layer_attributes['clip_scales'] * s_bound * self._global_clipping_coeff if layer_attributes['clip_scales'] is not None else None)
        self._l2_norm_scales.append(layer_attributes['l2_norm_scales'])
        self._effective_rank.append(layer_attributes['effective_rank'])

        layer.name = f"{key}_{layer_idx}"

        self._layers.append(layer)

    def key(self, key):
        return lambda layer, idx, s_bound: self.add_layer(layer, key, idx, s_bound)

    def _get_layer_attributes(self, network_cfg, layer_idx):
        attributes = ['clip_scales', 'l2_norm_scales', 'effective_rank', 'initialization']
        layer_attributes = [getattr(network_cfg, attr, None) for attr in attributes]

        for attribute in layer_attributes:
            if attribute is None:  
                return None
            elif isinstance(attribute, list): #check if omegaconf list
                if attribute[layer_idx - 1] != 0:
                    return attribute[layer_idx - 1]
                else:
                    return None
            else:
                return attribute
            
    def _initialize(self, layer, initialization):
        fan_in, fan_out = layer.weight.size(1), layer.weight.size(0)
        if initialization is None:
            s_bound = np.sqrt(2 / fan_in)
            nn.init.kaiming_uniform_(layer.weight)
        elif initialization == 'Xavier':
            s_bound = np.sqrt(6 / (fan_in + fan_out))
            nn.init.xavier_uniform_(layer.weight)
        return s_bound
    
    def clip_weights(self):
        for layer, clip in zip(self._layers, self._clip_scales):
            if clip is not None:
                layer.weight.data.clamp_(clip, -clip)

    def get_l2_norm_loss(self):
        l2_norm_loss = 0
        for layer, l2_norm_scale in zip(self._layers, self._l2_norm_scales):
            l2_norm = l2_norm_scale * (torch.sum(layer.weight ** 2) if l2_norm_scale else 0)
            l2_norm_loss += l2_norm
        return l2_norm_loss
    
    def get_effective_ranks(self):
        effective_ranks = []
        for layer, effective_rank in zip(self._layers, self._effective_rank):
            if effective_rank:
                rank = self._compute_effective_rank(layer.weight.data)
                effective_ranks.append({'name': layer.name, 'effective_rank': rank})
        return effective_ranks
    
    def _compute_effective_rank(matrix: torch.Tensor, delta: float = 0.01):
        """
        Computes the effective rank of a matrix based on the given delta value.
        Effective rank formula:
        srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
        See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
        """
        # Singular value decomposition. We only need the singular value matrix.
        _, S, _ = torch.linalg.svd(matrix)
        
        # Calculate the cumulative sum of singular values
        total_sum = S.sum()
        cumulative_sum = torch.cumsum(S, dim=0)
        
        # Find the smallest k that satisfies the effective rank condition
        threshold = (1 - delta) * total_sum
        effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
        
        return effective_rank

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    layer_norm=False,
    use_skip=False,
    transform_weights=None,
):
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []

    for i in range(1, len(sizes)):
        layer = nn.Linear(sizes[i - 1], sizes[i])
        layers.append(layer)
       
        if transform_weights is not None:
            transform_weights(layer, i)

        if i < len(sizes) - 1:
            layers.append(nonlinearity)
            if layer_norm:
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

