# Cortex

`cortex` provides an interface for a unified agent memory architecture.

## Architecture

Cortex implements a modular stack-based memory architecture with three core abstractions:

1. **Cells**: Stateless memory units (LSTM, GRU, etc.) that process sequences
   - Purpose: Encapsulate recurrent computation logic (gates, state updates, memory mechanisms)
   - Interface: Accepts input tensor and state, returns output and updated state
   - Examples: LSTM for long-term dependencies, GRU for simpler gated memory

2. **Blocks**: Wrappers around cells that handle projections and transformations
   - Purpose: Control information flow, stabilize gradients, and manage dimensionality

3. **Stacks**: Compositions of multiple blocks forming the complete memory system
   - Purpose: Build deep, hierarchical memory architectures

### Why This Design?

The separation of concerns between cells, blocks, and stacks provides several advantages:

- **Gradient Stability**: Blocks include skip connections, gating mechanisms, and normalization to prevent
  vanishing/exploding gradients in deep networks
- **Information Flow**: Learnable gates and skip paths allow the network to dynamically route information, bypassing
  cells when needed
- **Modularity**: Mix and match different memory cells without rewriting projection logic
- **Efficiency**: Blocks can operate cells at optimal dimensions (e.g., PreUp runs cells at 2x size for more capacity)
- **Flexibility**: Add new cell types or block patterns without modifying existing code
- **Clarity**: Clean separation between memory computation (cells) and architectural decisions (blocks/stacks)

### Key Features

- **Stateless design**: All state is explicitly passed as TensorDict inputs/outputs
- **Batch-first convention**: Consistent [B, T, H] tensor shapes throughout
- **Reset handling**: Per-timestep and per-batch reset masks for episode boundaries
- **Registry system**: Extensible architecture supporting custom cells and blocks
- **Type-safe configuration**: Pydantic-based configs with validation

## Quick Start

```python
from cortex import (
    CortexStackConfig,
    LSTMCellConfig,
    PreUpBlockConfig,
    PassThroughBlockConfig,
    build_cortex
)

# Define a memory stack configuration
config = CortexStackConfig(
    d_hidden=256,  # External hidden dimension
    blocks=[
        # First block: project up 2x, apply LSTM, project down
        PreUpBlockConfig(
            cell=LSTMCellConfig(hidden_size=512, num_layers=2),
            proj_factor=2.0
        ),
        # Second block: direct LSTM application
        PassThroughBlockConfig(
            cell=LSTMCellConfig(hidden_size=256, num_layers=1)
        )
    ],
    post_norm=True  # Apply LayerNorm after stack
)

# Build the stack
stack = build_cortex(config)

# Initialize state
batch_size = 4
state = stack.init_state(batch=batch_size, device="cuda", dtype=torch.float32)

# Forward pass with sequences
x = torch.randn(batch_size, seq_len, 256)  # [B, T, H]
output, new_state = stack(x, state)

# Handle resets per timestep
resets = torch.zeros(batch_size, seq_len, dtype=torch.bool)
resets[:, 5] = True  # Reset at timestep 5
output, new_state = stack(x, state, resets=resets)

# Single-step mode for inference
x_step = torch.randn(batch_size, 256)  # [B, H]
output_step, state = stack.step(x_step, state)
```

## Extending Cortex

### Custom Cell

```python
from cortex import MemoryCell, CellConfig, register_cell
from tensordict import TensorDict

@register_cell(MyCellConfig)
class MyCell(MemoryCell):
    def init_state(self, batch, *, device, dtype):
        return TensorDict({"hidden": torch.zeros(...)}, batch_size=[batch])

    def forward(self, x, state, *, resets=None):
        # Process input with state
        return output, new_state

    def reset_state(self, state, mask):
        # Apply reset mask to state
        return masked_state
```

### Custom Block

```python
from cortex import BaseBlock, BlockConfig, register_block

@register_block(MyBlockConfig)
class MyBlock(BaseBlock):
    def forward(self, x, state, *, resets=None):
        # Apply custom transformations around cell
        y, new_cell_state = self.cell(x, cell_state, resets=resets)
        return output, wrapped_state
```

Both custom cells and blocks are automatically available through the configuration system once registered.
