import hashlib
from torch import nn
import numpy as np
import torch

class MettaLayer(nn.Module):
    def __init__(self, layer, clip=None, l2_norm_scale=0.0):
        super(MettaLayer, self).__init__()
        self.layer = layer
        self.clip = clip
        self.l2_norm_scale = l2_norm_scale

    def forward(self, x):
        return self.layer(x)

    def clip_weights(self):
        if self.clip is not None:
            self.layer.weight.data.clamp_(self.clip, -self.clip)

    def l2_regularization(self):
        l2_norm = self.l2_norm_scale * (torch.sum(self.layer.weight ** 2) + (torch.sum(self.layer.bias ** 2) if self.layer.bias is not None else 0))
        return l2_norm

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    layer_norm=False,
    use_skip=False,
    global_clipping_value=None,
    clip_scales=None,  # List of scaling factors of global clipping value for each layer
    l2_norm_scales=None     # List of L2 coefficients for each layer
):
    """Create a stack of fully connected layers with nonlinearity, clipping, and L2 regularization."""
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []

    for i in range(1, len(sizes)):
        layer = nn.Linear(sizes[i - 1], sizes[i])
        clip = global_clipping_value * clip_scales[i - 1] if clip_scales and i - 1 < len(clip_scales) and global_clipping_value is not None else None
        l2_norm_scale = l2_norm_scales[i - 1] if l2_norm_scales and i - 1 < len(l2_norm_scales) else 0.0

        metta_layer = MettaLayer(layer, clip, l2_norm_scale)
        layers.append(metta_layer)

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

