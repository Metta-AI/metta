# Cortex

`cortex` is a modular memory stack library for building recurrent backbones and agent memory systems. It separates cell-level recurrence from architectural concerns (projections, skips, normalization) so you can compose new stacks quickly and safely.

## Architecture

Cortex implements a modular stack-based memory architecture with three core abstractions:

1. **Cells**: Stateless memory units (LSTM, GRU, etc.) that process sequences
   - Purpose: Encapsulate recurrent computation logic (gates, state updates, memory mechanisms)
   - Interface: Accepts input tensor and state, returns output and updated state
   - Examples: LSTM, mLSTM, sLSTM, AGaLiTe style memory, self-attention, or pretty much any other memory cell!

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

### Hidden Size Inference in Blocks

Some blocks define the working dimension of their nested cell. For those cases,
you may set the nested cell config's `hidden_size` to `None` and let the stack
builder infer it:

- `PreUpBlock`: infers `hidden_size = int(proj_factor * d_hidden)`.
- `PostUpBlock`: infers `hidden_size = d_hidden`.

This inference happens only when building via `CortexStackConfig`/`build_cortex`.
If you instantiate cells directly, provide a concrete `hidden_size`.

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
            cell=LSTMCellConfig(hidden_size=None, num_layers=2),  # inferred: 2.0 * d_hidden
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

## Template Architectures

The repo ships with a few small templates to get you started and to make quick comparisons easier.

- xlstm
  - Alternating `mLSTM` (PreUp) and `sLSTM` (PostUp) blocks.
  - Provided as a convenience builder `build_xlstm_stack(d_hidden, num_blocks, ...)`.

Example (xLSTM):

```python
from cortex.stacks.xlstm import build_xlstm_stack

stack = build_xlstm_stack(
    d_hidden=128,
    num_blocks=5,           # alternate mLSTM/sLSTM 5 times
    mlstm_proj_factor=2.0,  # PreUp factor for mLSTM blocks
    slstm_proj_factor=1.5,  # PostUp factor for sLSTM blocks
)
```

You can also run these templates in the synthetic evaluation harness (see “Evaluate Quickly”).

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

### Create a New Architecture (Stack Recipe)

Follow this process to add a new architecture:

1) Pick a block pattern and sizes
- Decide how many blocks you want and whether each should be `PreUp`, `PostUp`, or `PassThrough`.
- Choose `proj_factor` for `PreUp`/`PostUp` blocks; `d_hidden` is the fixed external width.

2) Write a builder in `cortex/stacks/`
- Create `packages/cortex/src/cortex/stacks/my_arch.py` with a small helper that returns a `CortexStack`.

```python
from cortex.config import CortexStackConfig, PreUpBlockConfig, PostUpBlockConfig, mLSTMCellConfig, sLSTMCellConfig
from cortex.stacks.base import CortexStack

def build_my_arch(d_hidden: int, *, num_blocks: int = 4) -> CortexStack:
    blocks = [
        PreUpBlockConfig(cell=mLSTMCellConfig(hidden_size=None, num_heads=4), proj_factor=2.0),
        PostUpBlockConfig(cell=sLSTMCellConfig(hidden_size=None, num_heads=4), proj_factor=1.5),
        # ...repeat or vary as needed...
    ]
    cfg = CortexStackConfig(d_hidden=d_hidden, blocks=blocks, post_norm=True)
    return CortexStack(cfg)
```

3) Export it
- Add the builder to `packages/cortex/src/cortex/stacks/__init__.py` so users can import it.

```python
from cortex.stacks.my_arch import build_my_arch  # and add to __all__
```

4) Register a template for quick evals (optional, but recommended)
- Edit `packages/cortex/evaluations/stacks.py` and register your builder under `STACKS`:

```python
from cortex.stacks.my_arch import build_my_arch
STACKS["my_arch"] = StackSpec(name="my_arch", builder=lambda: build_my_arch(d_hidden=128), d_hidden=128)
```

5) Evaluate quickly (CLI)
- Run the synthetic tasks to sanity‑check wiring and step/sequence parity:

```bash
python packages/cortex/evaluations/run.py --task delayed_recall --stack my_arch
python packages/cortex/evaluations/run.py --task majority --stack all   # runs all registered templates
```

6) (Optional) Add tests
- See `packages/cortex/tests/test_cortex_stack.py` for examples that check shapes, state handling, and resets.


### Tips
- Prefer batch‑first shapes `[B, T, H]` and pass state explicitly.
- Use `PreUpBlock` when a cell benefits from a larger inner width; use `PostUpBlock` to stabilize depth with a cell at `d_hidden`.
- Let the stack infer `cell.hidden_size=None` inside `PreUpBlock`/`PostUpBlock` unless you’re composing blocks manually.
