# Cortex

`cortex` is a modular library for building recurrent backbones and agent memory systems. It separates cell-level recurrence from architectural concerns (projections, skips, normalization) so you can compose new stacks quickly and safely.

## Table of Contents

- [Architecture](#architecture)
  - [Why This Design?](#why-this-design)
  - [Uniform Interface Design](#uniform-interface-design)
- [Supported Components](#supported-components)
  - [Memory Cells](#memory-cells)
  - [Blocks](#blocks)
- [Axons (Streaming RTU)](#axons-streaming-rtu)
- [Quick Start](#quick-start)
- [Template Architectures](#template-architectures)
- [Metta Framework Integration](#metta-framework-integration)
- [Evaluate Quickly](#evaluate-quickly)
- [Backend Configuration](#backend-configuration)
- [Extending Cortex](#extending-cortex)
  - [Custom Cell](#custom-cell)
  - [Custom Block](#custom-block)
  - [Create a New Architecture (Stack Recipe)](#create-a-new-architecture-stack-recipe)

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

- **Modularity**: Mix and match different memory cells without rewriting projection logic
- **Efficiency**: GPU-accelerated Triton kernels provide optimized implementations for core cells (LSTM, mLSTM, sLSTM,
  CausalConv1d) with automatic fallback to PyTorch. Triton kernels deliver significant speedups on CUDA while
  maintaining numerical parity with reference implementations.
- **Composability**: The design is fully compositional—cells can be built from other cells, blocks can contain multiple
  cells or even nest other blocks, and stacks can be composed of other stacks. This recursive structure enables
  arbitrarily complex architectures while maintaining clean interfaces.
- **Auto-Configuration**: Cells and blocks are automatically configured with hidden sizes inferred from the highest level
  of abstraction (the stack's `d_hidden`). This top-down configuration makes it easy to define complex architectures
  without manual dimension tracking.
- **Flexibility**: Add new cell types or block patterns without modifying existing code
- **Clarity**: Clean separation between memory computation (cells) and architectural decisions (blocks/stacks)
- **Gradient Stability**: Blocks include skip connections, gating mechanisms, and normalization to prevent
  vanishing/exploding gradients in deep networks
- **Information Flow**: Learnable gates and skip paths allow the network to dynamically route information, bypassing
  cells when needed

### Uniform Interface Design

A core principle of Cortex is that **cells, blocks, and stacks all share the same interface**, enabling seamless composition at any level:

**Shared Signatures:**
```python
# All three abstractions implement these methods:
def forward(x: Tensor, state: TensorDict, *, resets: Optional[ResetMask] = None) -> Tuple[Tensor, TensorDict]:
    """Process input with state, optionally applying resets, return output and new state."""

def init_state(batch: int, *, device: torch.device, dtype: torch.dtype) -> TensorDict:
    """Initialize state for a batch."""

def reset_state(state: TensorDict, mask: ResetMask) -> TensorDict:
    """Apply episode boundary resets to state (rarely needed, see below)."""
```

**Key Properties:**
- **Consistent shapes**: All accept `[B, T, H]` for sequences or `[B, H]` for single-step
- **TensorDict state**: State is always a TensorDict with arbitrary nesting depth
  - Cells: Flat state (e.g., `{"h": ..., "c": ...}`)
  - Blocks: Nest cell state under cell class name (e.g., `{"LSTMCell": {"h": ..., "c": ...}}`)
  - Stacks: Nest block states under indexed keys (e.g., `{"PreUpBlock_0": {"LSTMCell": {...}}}`)
- **Automatic reset handling**: Resets are handled automatically when passed through `forward(resets=mask)`
  - The reset mask propagates through Stack → Block → Cell automatically
  - `reset_state()` exists for completeness but is typically not needed in practice
  - Just pass `resets` to `forward()` and the hierarchy handles it internally

This uniformity means you can treat a complex multi-layer stack exactly like a single cell, enabling arbitrary composition without changing your code interface.


## Supported Components

### Memory Cells

Core computational units implementing recurrent logic. All cells follow batch-first convention: `[B, T, H]` for sequences, `[B, H]` for single-step.

| Cell            | Description | Triton Accelerated | CUDA Accelerated |
|-----------------|-------------|--------------------|------------------|
| `LSTMCell`      | Stateless LSTM wrapper with TensorDict state (`h`, `c`); step and sequence parity; optional resets. | Yes | No |
| `mLSTMCell`     | Matrix-LSTM with per-head state, chunkwise closed-form updates, and optional causal Conv1D pre-activation. | Yes | No |
| `sLSTMCell`     | Structured LSTM with per-head gating, stabilized accumulators (`c`, `n`, `m`), and optional causal Conv1D. | Yes | No |
| `CausalConv1d`  | Depthwise causal Conv1D cell (ring-buffer state); supports optional channel-mixing mode. | Yes (channel-mixing only) | No |
| `Axons`         | Streaming RTU with diagonal input weights (per-channel local recurrence, 2H→H projection). | Yes | Yes (seq‑allin, short‑T) |

**Notes:**
- Triton kernels are selected automatically on CUDA when constraints are met; otherwise PyTorch fallback is used
- `CausalConv1d` uses Triton only in channel-mixing mode (groups=1) with per-timestep resets
- Resets (episode boundaries) are optional and broadcast-safe: `[B, T]` for sequences, `[B]` for steps

### Blocks

Wrappers around cells that handle projections, normalization, and information flow.

| Block              | Description |
|--------------------|------------|
| `PassThroughBlock` | Applies the nested cell directly at `d_hidden` with residual; no projections. |
| `PreUpBlock`       | Pre-upsamples to `d_inner = int(proj_factor * d_hidden)`, runs the cell at `d_inner`, gates and projects back to `d_hidden`, then adds residual. |
| `PostUpBlock`      | Runs the cell at `d_hidden`, then applies a gated feed-forward projection up and back down before residual. Useful for deep stacks. |
| `AdapterBlock`     | Wraps another block with a trainable residual adapter (identity at init). Lets you insert capacity without changing behavior at t=0. |

#### Hidden Size Inference in Blocks

**Important**: Some blocks control the working dimension of their nested cell and will **override** the cell's `hidden_size` during stack construction, regardless of what value you provide:

- `PreUpBlock`: Sets `cell.hidden_size = int(proj_factor * d_hidden)`
- `PostUpBlock`: Sets `cell.hidden_size = d_hidden`
- `PassThroughBlock`: Sets `cell.hidden_size = d_hidden`

**Best Practice**: Set `hidden_size = None` in the cell config to make the override explicit and avoid confusion. The builder will infer and set the correct value automatically.

```python
# Recommended: explicit None shows the value will be inferred
PreUpBlockConfig(
    cell=LSTMCellConfig(hidden_size=None),  # Will be set to int(2.0 * 256) = 512
    proj_factor=2.0
)

# Also works but misleading: 128 will be ignored and overridden to 512
PreUpBlockConfig(
    cell=LSTMCellConfig(hidden_size=128),  # Ignored! Actually becomes 512
    proj_factor=2.0
)
```

This override happens only when building via `CortexStackConfig`/`build_cortex`. If you instantiate blocks and cells directly (without the stack builder), you must provide concrete sizes that satisfy these relationships manually.

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

## Axons - Locally Recurrent Gradient Propagating alternative to Linear Layers

The Axons cell (`packages/cortex/src/cortex/cells/axons.py`) is the new
fundamental building block in Cortex. It is designed as a drop‑in replacement
for a linear layer that makes the projection locally recurrent and enables
gradients to propagate across an arbitrary horizon.

Key properties

- Linear→Recurrent: Wraps a diagonal RTU update around each channel and returns
  a 2H activation that is projected back to H. This effectively turns a
  `Linear(H→H)` into a locally recurrent layer with minimal parameter overhead.
- Optional SRHT mixer: Before the kernel, Axons can apply a fast Subsampled
  Randomized Hadamard Transform (SRHT) that orthonormally mixes features along
  H. This improves conditioning and lets diagonal per‑channel weights behave
  more like a dense map. Implementations:
  - CUDA FWHT path (power‑of‑two H), with custom backward.
  - PyTorch FWHT fallback (any H) with autograd support.
  Configure via `AxonsConfig.use_srht` (default True; set False to disable) and `AxonsConfig.srht_permute`
  (random permutation on/off). The SRHT is normalized by 1/√H, so it preserves
  feature scale and gradient norms.
- Streaming traces: Carries compact eligibility traces across subsequences and
  applies a single boundary correction at chunk heads. This preserves
  cross‑chunk credit assignment, allowing learning beyond the TBPTT truncation
  length without storing full activations.
- Backend auto‑selection: Three aligned implementations are provided and chosen
  at runtime:
  - CUDA short‑T fused sequential kernel — preferred when `T ≤ cuda_seq_threshold`.
  - Triton streaming kernel (GPU, long‑T fast path).
  - PyTorch reference (portable baseline).
- Tunable policy: Control the short‑T cutoff via
  `AxonsConfig.cuda_seq_threshold` (default 1000).

Usage example

```python
import torch
from cortex.config import AxonsConfig
from cortex.cells.axons import Axons

B, T, H = 8, 512, 256
x = torch.randn(B, T, H, device='cuda')

cell = Axons(
    AxonsConfig(
        hidden_size=H,
        activation='SiLU',
        cuda_seq_threshold=1000,
        # use_srht is True by default; set to False to disable
        use_srht=True,
        srht_permute=True       # random permutation in SRHT
    )
).to(x.device)
state = cell.init_state(batch=B, device=x.device, dtype=x.dtype)

# Optional per‑timestep resets: shape [B, T] (bool)
resets = torch.zeros(B, T, dtype=torch.bool, device=x.device)

y, state = cell(x, state=state, resets=resets)  # y: [B, T, H]
```

Why it matters

- Axons generalizes the ubiquitous linear layer to be locally recurrent and
  streaming, improving temporal expressivity with low memory cost.
- It enables learning signals to flow beyond TBPTT by preserving
  cross‑boundary credit via compact traces rather than full activation history.
- It is intended as the default “linear” building block in future Cortex
  architectures: replacing feed‑forward linear projections with Axons imbues
  networks with long‑horizon temporal inductive bias.

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


## Metta Framework Integration

Cortex provides a ready-to-use adapter for integrating memory stacks with the [Metta RL framework](https://github.com/metta-ai/metta), which uses TensorDict-based state management.

### MettaTDAdapter

The `MettaTDAdapter` wraps a `CortexStack` and makes it compatible with Metta's TensorDict interface, handling stateful recurrent memory across rollout and training phases.

**Key Features:**

- **Two-cache architecture**: Separate caches for rollout (data collection) and training (gradient updates)
  - `rollout_cache`: Updated during environment interaction (single-step mode)
  - `train_cache`: Frozen snapshot used for deterministic training on replayed sequences
- **Per-environment memory**: Efficiently manages separate hidden states for each environment
- **Reset handling**: Automatically applies episode boundary resets via done/truncated flags
- **Memory-efficient replay**: Stores only `env_id` in replay buffer; reconstructs states from cache
- **Optional output projection**: Configurable linear projection after the stack

**Example Usage:**

```python
from cortex import build_cortex, CortexStackConfig, LSTMCellConfig, PreUpBlockConfig
from cortex.adapters import MettaTDAdapter

# Build a memory stack
stack = build_cortex(CortexStackConfig(
    d_hidden=256,
    blocks=[
        PreUpBlockConfig(
            cell=LSTMCellConfig(hidden_size=None),
            proj_factor=2.0
        )
    ]
))

# Wrap with Metta adapter
adapter = MettaTDAdapter(
    stack=stack,
    in_key="latent",           # Input key in TensorDict
    out_key="recurrent_out",   # Output key in TensorDict
    d_hidden=256,              # Stack's external hidden size
    out_features=512,          # Optional projection to different size
    store_dtype="fp32"         # Storage precision: 'fp32' or 'bf16'
)

# Use in your Metta policy
# The adapter handles state management automatically via TensorDict metadata
```

**Integration Notes:**

- The adapter is an `nn.Module` that registers the stack's parameters for optimization
- Requires `training_env_ids` in TensorDict for per-environment state tracking
- Expects `bptt` (backprop through time steps) metadata to distinguish rollout (bptt=1) from training (bptt>1)
- Implements `get_memory()` / `set_memory()` for checkpoint serialization
- Reset masks are constructed from `dones` and `truncateds` in the TensorDict

**Performance Considerations:**

- Uses `store_dtype` to control memory vs precision tradeoff (fp32 for accuracy, bf16 for memory)
- Maintains a persistent batched state during rollout to avoid redundant gather/scatter operations
- Training cache is compacted to only store active environment states


## Evaluate Quickly

Cortex includes lightweight synthetic tasks for sanity-checking stacks and comparing recipes. These evaluations verify wiring, state handling, and step/sequence parity.

**Available Tasks:**
- `delayed_recall` (T-Maze): Tests long-range memory retention
- `majority`: Tests additive integration and counting
- `dyck`: Tests stack-like behavior with balanced parentheses

**Run a Single Stack:**

```bash
uv run python packages/cortex/evaluations/run.py --task delayed_recall --stack slstm_postup
```

**Run All Registered Stacks:**

```bash
uv run python packages/cortex/evaluations/run.py --task majority --stack all
```

**Common Flags:**
- `--epochs`, `--batch-size`, `--lr`, `--seed`
- `--log-level {DEBUG, INFO, WARNING, ERROR}`

For more details, see [`docs/evaluations.md`](docs/evaluations.md).

## Backend Configuration

Cortex automatically selects between Triton (GPU-accelerated) and PyTorch backends based on device, dtype, and cell constraints. You can override this behavior with environment variables:

```bash
# Force PyTorch backend (disable Triton)
CORTEX_DISABLE_TRITON=1 python your_script.py

# Equivalent alternative
CORTEX_FORCE_PYTORCH=1 python your_script.py
```

**When to Use:**
- Debugging numerical differences between backends
- Testing on systems without Triton support
- Ensuring consistent behavior during development

**Backend Selection Details:**
- Triton kernels are used automatically on CUDA for supported cells (`LSTMCell`, `mLSTMCell`, `sLSTMCell`, `CausalConv1d`)
- Some cells have additional constraints (e.g., LSTM requires power-of-two `hidden_size`)
- Falls back to PyTorch when constraints aren't met

For more details, see [`docs/api/kernels.md`](docs/api/kernels.md).

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
- Let the stack infer `cell.hidden_size=None` inside `PreUpBlock`/`PostUpBlock` unless you're composing blocks manually.
