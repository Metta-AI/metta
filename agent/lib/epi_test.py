import torch
import torch.nn as nn
from agent.lib.util import initialize_weights, epi_initialize

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
    epi_initialize(model[2], row_specs={2: 1.0, 4: -0.8})

    # Create random inputs
    x = torch.randn(10, 128)

    # Forward pass
    y = model(x)

    # Print the outputs
    print("Network outputs:")
    print(y)

if __name__ == "__main__":
    test_epi_init()