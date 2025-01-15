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

def initialize_weights(layer, method='xavier', epi_row_specs=None, nonlinearity=None):
    if method == 'xavier':
        init.xavier_uniform_(layer.weight.data)
    elif method == 'normal':
        init.normal_(layer.weight.data, mean=0.0, std=0.02)
    elif method == 'he':
        init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
    elif method == 'orthogonal':
        init.orthogonal_(layer.weight.data)
    elif method == 'epi':
        epi_initialize_rows(layer, epi_row_specs, nonlinearity)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
def append_nonlinearity(layers, nonlinearity):
    if nonlinearity is not None:
        if nonlinearity == 'tanh':
            layers.append(nn.Tanh())
        elif nonlinearity == 'relu':
            layers.append(nn.ReLU())
        elif nonlinearity == 'elu':
            layers.append(nn.ELU())
        elif nonlinearity == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif nonlinearity == 'softmax':
            layers.append(nn.Softmax(dim=1))
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity=nn.ELU(),
    initialization=None,
    # bias_initialization=None,
    layer_norm=False,
    use_skip=False,
    epi_init=False,
    epi_row_specs=None,
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
            # layers.append(nonlinearity)
            append_nonlinearity(layers, nonlinearity)

        if layer_norm and i < len(sizes) - 1:
            layers.append(nn.LayerNorm(sizes[i]))

    if epi_init:
        initialize_weights(layers[-1], method='epi', epi_row_specs=epi_row_specs, nonlinearity=nonlinearity)

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

def epi_initialize_rows(layer: nn.Linear, epi_row_specs=None, nonlinearity='tanh'):
    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(nonlinearity))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    out_features, in_features = layer.weight.shape
    a = math.sqrt(6.0 / (in_features + out_features))
    print(f"Xavier bound: {a}")

    with torch.no_grad():
        for row_idx, mag in epi_row_specs.items():
            layer.weight[row_idx, :].fill_(mag * a)

        for row_idx in range(out_features):
            if row_idx not in epi_row_specs:
                layer.weight[row_idx, :].uniform_(-a, a)

        # orthogonalize
        W_t = layer.weight.data.t()
        Q, R = torch.linalg.qr(W_t, mode='reduced')
        W_ortho = torch.mm(R.t(), Q.t())        # print(f"pre copy layer.weight row 2: {layer.weight[2, :6]}")
        layer.weight.detach().copy_(W_ortho)
        print(f"post copy layer.weight row 0: {layer.weight[0, :6]}")
        print(f"post copy layer.weight row 1: {layer.weight[1, :6]}")
        print(f"post copy layer.weight row 2: {layer.weight[2, :6]}")

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
    model = nn.Sequential(
        nn.Linear(128, 1028),
        nn.ReLU(),
        nn.Linear(1028, 20),
    )

    # 1) Orthogonal init for the first layer
    initialize_weights(model[0], method='orthogonal')

    # 2) epi_initialize for the second layer
    epi_initialize_rows(model[2], 
                        {1: 0.9, 7: 0.2, 8: -0.8, 9: -0.8, 10: -0.8, 11: -0.8, 12: -0.8, 13: -0.8, 14: -0.8, 15: -0.8, 16: -0.8, 17: -0.8}, 
                        init_nonlinearity='tanh'
    )

    # Create random inputs
    x = torch.randn(10, 128)

    # Forward pass
    y = model(x)

    # Print the outputs
    print("Network outputs:")
    torch.set_printoptions(sci_mode=False)
    print(y)

    random_inputs = torch.randn(10, 128)
    for idx, input_vector in enumerate(random_inputs):
        output = model(input_vector)
        print(f"Output for input vector {idx + 1}:\n{output}\n")

if __name__ == "__main__":
    test_epi_init()