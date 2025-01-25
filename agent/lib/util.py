import hashlib
from torch import nn
import numpy as np
import torch

class WeightTransformer():
    def __init__(self, cfg):
        self._clip_scales = []
        self._l2_norm_scales = []
        self._layers = []
        self.cfg = cfg
        self._global_clipping_value = cfg.clipping_value

    def _get_scale(self, network_cfg, attr_name, layer_idx):
        scales = getattr(network_cfg, attr_name, None)
        if scales is not None and layer_idx - 1 < len(scales):
            return list(scales)[layer_idx - 1]
        return None

    def add_layer(self, layer, key, layer_idx):
        clip_scale = self._get_scale(self.cfg.get(key), 'clip_scales', layer_idx)
        l2_norm_scale = self._get_scale(self.cfg.get(key), 'l2_norm_scales', layer_idx)

        self._clip_scales.append(clip_scale)
        self._l2_norm_scales.append(l2_norm_scale)
        self._layers.append(layer)

    def key(self, key):
        return lambda layer, idx: self.add_layer(layer, key, idx)
    
    def clip_weights(self):
        for layer, clip in zip(self._layers, self._clip_scales):
            if clip is not None:
                layer.weight.data.clamp_(clip, -clip)

    def get_l2_norm_loss(self):
        l2_norm = self.l2_norm_scale * (torch.sum(self.layer.weight ** 2) if self.l2_norm_scale else 0)
        return l2_norm

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    layer_norm=False,
    use_skip=False,
    transform_weights=None,
):
    """Create a stack of fully connected layers with nonlinearity, clipping, and L2 regularization."""
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

