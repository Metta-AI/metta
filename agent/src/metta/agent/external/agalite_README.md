# AGaLiTe (Approximate Gated Linear Transformer) PyTorch Implementation

## Overview

This is a PyTorch implementation of AGaLiTe, converted from the original JAX implementation. AGaLiTe is a recurrent transformer architecture designed for partially observable online reinforcement learning, offering context-independent inference cost while leveraging long-range dependencies effectively.

## Key Components

### 1. **GRU Gating Unit**
- Controls information flow through the network
- Initialized close to identity map for stable learning
- Used to combine attention outputs with previous states

### 2. **Parameterized Projection**
- Creates feature maps through learned outer products
- Replaces fixed kernel functions with trainable projections
- Enables the model to learn complex non-linear relationships

### 3. **AGaLiTe Attention Layer**
- Core attention mechanism with approximation using cosine basis functions
- Uses r approximation terms to represent the attention state
- Features:
  - Gated updates with beta and gamma parameters
  - Feature-mapped keys and queries using outer products
  - Oscillatory encoding for temporal information
  - Discounted sum for maintaining history

### 4. **Recurrent Linear Transformer Encoder**
- Single encoder layer combining:
  - Layer normalization
  - AGaLiTe attention
  - GRU gating units
  - Feed-forward network
- First layer uses optional dense embedding

### 5. **AGaLiTe Model**
- Stacks multiple encoder layers
- Maintains separate memory for each layer
- Supports termination signals for episode boundaries

### 6. **BatchedAGaLiTe**
- Processes multiple sequences in parallel
- Handles batched memory states
- Efficient for training with multiple environments

## Key Parameters

- `n_layers`: Number of transformer layers
- `d_model`: Model dimension
- `d_head`: Dimension per attention head
- `n_heads`: Number of attention heads
- `eta`: Feature map expansion factor (controls memory capacity)
- `r`: Number of approximation terms (accuracy vs. efficiency trade-off)
- `reset_on_terminate`: Whether to reset hidden states on episode termination

## Memory Management

The model maintains a recurrent state consisting of:
- `tilde_k`: Approximated key states (r, n_heads, eta * d_head)
- `tilde_v`: Approximated value states (r, n_heads, d_head)
- `s`: Normalization states (n_heads, eta * d_head)
- `tick`: Time counter for oscillatory encoding

## Usage Example

```python
import torch
from agalite import AGaLiTe

# Initialize model
model = AGaLiTe(
    n_layers=4,
    d_model=256,
    d_head=64,
    d_ffc=512,
    n_heads=4,
    eta=4,  # Feature map expansion
    r=7,    # Number of approximation terms
    reset_on_terminate=True
)

# Initialize memory
memory = AGaLiTe.initialize_memory(
    n_layers=4,
    n_heads=4,
    d_head=64,
    eta=4,
    r=7
)

# Forward pass
inputs = torch.randn(100, 256)  # (sequence_length, d_model)
terminations = torch.zeros(100)  # Episode termination signals

outputs, new_memory = model(inputs, terminations, memory)
```

## Computational Efficiency

AGaLiTe offers significant computational advantages over standard transformers:
- **Context-independent inference cost**: O(rηd) space and time complexity
- **40% cheaper inference** compared to GTrXL
- **50% less memory usage** compared to GTrXL
- Scales well with sequence length without quadratic complexity

## Differences from JAX Implementation

This PyTorch implementation maintains algorithmic equivalence while adapting to PyTorch conventions:
1. Uses PyTorch's `nn.Module` instead of Flax's `linen`
2. Replaces JAX's `scan` operations with explicit loops
3. Adapts initialization methods to PyTorch's approach
4. Uses PyTorch's autograd for gradient computation

## References

- Paper: "AGaLiTe: Approximate Gated Linear Transformers for Online Reinforcement Learning"
- Original JAX implementation: https://github.com/subho406/agalite

## Integration with Metta

This implementation is designed to be compatible with the Metta RL framework and can be used as a drop-in replacement for other recurrent architectures in partially observable environments.