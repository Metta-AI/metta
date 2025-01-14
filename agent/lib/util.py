import hashlib
from torch import nn
import numpy as np
import torch
import torch.nn.init as init

'''
todo
ensure that removal of bias initialization in metta agent is okay
adjust epi to allow for various vectors and magnitudes
get launch to read from simple.matters.yaml
'''

def initialize_weights(layer, method='xavier'):
    if method == 'xavier':
        init.xavier_uniform_(layer.weight.data)
    elif method == 'normal':
        init.normal_(layer.weight.data, mean=0.0, std=0.02)
    elif method == 'he':
        init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
    elif method == 'orthogonal':
        init.orthogonal_(layer.weight.data)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    initialization=None,
    # bias_initialization=None,
    layer_norm=False,
    use_skip=False,
):
    """Create a stack of fully connected layers with nonlinearity"""
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        if initialization:
            initialize_weights(layers[-1], method=initialization)
        # if bias_initialization:
        #     layers[-1].bias.data = bias_initialization(layers[-1].bias.data)

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
    
import math
import torch
import torch.nn as nn

def epi_initialize_rows(layer: nn.Linear, row_specs=None):
    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    out_features, in_features = layer.weight.shape
    a = math.sqrt(6.0 / (in_features + out_features))
    a = 1.0 # delete this
    print(f"Xavier bound: {a}")

    with torch.no_grad():
        for row_idx, mag in row_specs.items():
            layer.weight[row_idx, :].fill_(mag * a)

        for row_idx in range(out_features):
            if row_idx not in row_specs:
                layer.weight[row_idx, :].uniform_(-a, a)

        # orthogonalize
        W_t = layer.weight.data.t()
        Q, R = torch.linalg.qr(W_t, mode='reduced')
        W_ortho = torch.mm(R.t(), Q.t())
        print(f"W_ortho row 0: {W_ortho[0, :]}")
        print(f"layer.weight row 0: {layer.weight[0, :]}")
        print(f"W_ortho row 2: {W_ortho[2, :]}")
        print(f"pre copy layer.weight row 2: {layer.weight[2, :]}")
        layer.weight.detach().copy_(W_ortho)
        print(f"post copy layer.weight row 0: {layer.weight[0, :]}")
        layer.weight.detach().copy_(W_ortho)

# def epi_initialize(layer: nn.Linear, row_specs=None):
#     """
#     Initializes `layer.weight` (shape: [out_features, in_features]) so that:
#       1. Compute the Xavier (Glorot) bound via:
#            a = sqrt(6.0 / (fan_in + fan_out))
#       2. For each (row_index -> magnitude) in `row_specs`, fill that row
#          with (magnitude * a).
#       3. Fill all other rows uniformly in [-a, a].
#       4. Orthogonalize the result by QR factorization on W^T => W = R^T Q^T.

#     Args:
#         layer (nn.Linear): The linear layer whose weights we are initializing.
#         row_specs (dict, optional): A dictionary of {row_index: magnitude}, where
#                                     magnitude ∈ [-1, 1]. Defaults to None.
#         a (float, optional): If provided, use this as the bound. Otherwise,
#                              compute via Xavier initialization. Defaults to None.

#     Note:
#         Perfect orthogonality might be compromised by locking certain rows, but
#         we at least try to preserve as much orthogonality as possible.
#     """
#     if row_specs is None:
#         row_specs = {}  # by default, no pinned rows

#     out_features, in_features = layer.weight.shape

#     # Compute the Xavier (Glorot) bound
#     a = math.sqrt(6.0 / (in_features + out_features))

#     # Validate user-specified rows
#     for row_idx, mag in row_specs.items():
#         if row_idx < 0 or row_idx >= out_features:
#             raise ValueError(f"Row index {row_idx} is out of range [0, {out_features-1}].")
#         if not (-1.0 <= mag <= 1.0):
#             raise ValueError(f"Row {row_idx} magnitude must be in [-1, 1], got {mag}.")

#     with torch.no_grad():
#         W = layer.weight  # shape: [out_features, in_features]

#         # 1) Fill pinned rows
#         for row_idx, mag in row_specs.items():
#             W[row_idx, :].fill_(mag * a)

#         # 2) Fill other rows in [-a, a]
#         for row_idx in range(out_features):
#             if row_idx not in row_specs:
#                 W[row_idx, :].uniform_(-a, a)

#         # 3) Orthogonalize (QR factorization)
#         W_t = W.data.t()             # shape: [in_features, out_features]
#         Q, R = torch.qr(W_t)         # Q: [in_features, in_features], R: [in_features, out_features]
#         W_ortho = torch.mm(R.t(), Q.t())  # shape: [out_features, in_features]

#         # 4) Copy back
#         W.copy_(W_ortho)

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

def test_epi_init():
    # Build a small sequential net:
    # Input: 128 -> Hidden: 512 -> Output: 20
    # Use tanh nonlinearity, and custom initialization as specified.
    model = nn.Sequential(
        nn.Linear(128, 512),
        nn.Tanh(),
        nn.Linear(512, 20),
    )

    # 1) Orthogonal init for the first layer
    initialize_weights(model[0], method='orthogonal')

    # 2) epi_initialize for the second layer
    epi_initialize_rows(model[2], row_specs={0: 0.9, 2: -0.8})

    # Create random inputs
    x = torch.randn(10, 128)

    # Forward pass
    y = model(x)

    # Print the outputs
    print("Network outputs:")
    print(y)

if __name__ == "__main__":
    test_epi_init()