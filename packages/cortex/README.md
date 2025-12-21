# Cortex

`cortex` is a modular library for building recurrent backbones and agent memory systems. It separates cell-level
recurrence from architectural concerns (projections, skips, normalization) so you can compose new stacks quickly and
safely.

## Install

Use the cortexcore distribution; import path remains `cortex`:

```
pip install cortexcore
```

## Table of Contents

- [Install](#install)
- [Architecture](#architecture)
  - [Why This Design?](#why-this-design)
  - [Uniform Interface Design](#uniform-interface-design)
- [Quick Start](#quick-start)
- [Supported Components](#supported-components)
  - [Memory Cells](#memory-cells)
  - [Blocks](#blocks)
- [MoE using Column](#moe-using-column)
- [Advanced Setup](#advanced-setup)
- [Metta Framework Integration](#metta-framework-integration)
- [AxonLayer: A Generalized Linear Operator with Stateful Dynamics](#axonlayer-a-generalized-linear-operator-with-stateful-dynamics)
  - [AxonLayer Integration Across Cells](#axonlayer-integration-across-cells)
- [Evaluate Quickly](#evaluate-quickly)
- [Backend Configuration](#backend-configuration)
- [Extending Cortex](#extending-cortex)
  - [Custom Cell](#custom-cell)
  - [Custom Block](#custom-block)
  - [Create a New Architecture (Stack Recipe)](#create-a-new-architecture-stack-recipe)

## Architecture

Cortex implements a modular stack-based memory architecture with four core abstractions:

1. **Cells**: Stateless memory units (LSTM, GRU, etc.) that process sequences
   - Purpose: Encapsulate recurrent computation logic (gates, state updates, memory mechanisms)
   - Interface: Accepts input tensor and state, returns output and updated state
   - Examples: LSTM, mLSTM, sLSTM, AGaLiTe style memory, self-attention, or pretty much any other memory cell!

2. **Blocks**: Wrappers around cells that handle projections and transformations
   - Purpose: Control information flow, stabilize gradients, and manage dimensionality

3. **Column**: A router‑mixed set of expert blocks executed in parallel and combined
   - Purpose: Let multiple block "experts" compete/cooperate per token/over time.
   - How: A global prior gate (with optional per‑token refinement) mixes expert deltas; an E‑axis mixer and outer ReZero
     stabilize depth
   - Code: `packages/cortex/src/cortex/blocks/column/column.py` and helpers in
     `packages/cortex/src/cortex/blocks/column/auto.py`

4. **Stacks**: Compositions of multiple blocks forming the complete memory system
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
- **Auto-Configuration**: Cells and blocks are automatically configured with hidden sizes inferred from the highest
  level of abstraction (the stack's `d_hidden`). This top-down configuration makes it easy to define complex
  architectures without manual dimension tracking.
- **Flexibility**: Add new cell types or block patterns without modifying existing code
- **Clarity**: Clean separation between memory computation (cells) and architectural decisions (blocks/stacks)
- **Gradient Stability**: Blocks include skip connections, gating mechanisms, and normalization to prevent
  vanishing/exploding gradients in deep networks
- **Information Flow**: Learnable gates and skip paths allow the network to dynamically route information, bypassing
  cells when needed

### Uniform Interface Design

A core principle of Cortex is that **cells, blocks, and stacks all share the same interface**, enabling seamless
composition at any level:

**Shared Signatures:**

```python
# All four abstractions implement a unified interface for forward pass
def forward(x: Tensor, state: TensorDict, *, resets: Optional[ResetMask] = None) -> Tuple[Tensor, TensorDict]:
    """Process input with state, optionally applying resets, return output and new state."""
    ...
    return out, new_state
```

- **Input (x)**: All accept `[B, T, H]` for sequences or `[B, H]` for single-step
- **Recurrent state**: State is always a TensorDict with arbitrary nesting depth
  - Cells: Flat state (e.g., `{"h": ..., "c": ...}`)
  - Blocks: Nest cell state under cell class name (e.g., `{"LSTMCell": {"h": ..., "c": ...}}`)
  - Columns: One entry per expert (e.g., `{"expert_PreUpBlock_0": {...}, ...}`)
  - Stacks: Nest block states under indexed keys (e.g., `{"PreUpBlock_0": {"LSTMCell": {...}}}`)
- **Output (out)**: `[B, T, H]` for sequences or `[B, H]` for single-step
- **Automatic reset handling**: Resets are handled automatically when passed through `forward(resets=mask)`
  - The reset mask propagates through Stack → Block → Cell automatically

This uniformity means you can treat a complex multi-layer stack exactly like a single cell, enabling arbitrary
composition without changing your code interface.

## Quick Start

Use the auto stack DSL in `packages/cortex/src/cortex/stacks/auto.py`, which builds a stack of Column layers from
compact patterns of expert tokens. Each layer is a Column whose experts are chosen by a pattern such as "AXMS".

Built‑in expert tokens:

- `A` = Axon (PostUp)
- `X` = Transformer‑XL (PostUp, GRU‑gated)
- `M` = mLSTM (PreUp)
- `S` = sLSTM (PostUp)
- Suffix `^` enables Axon projections for that expert where supported (e.g., `M^`, `X^`, `S^`).

```python
import torch
from cortex.stacks import build_cortex_auto_stack  # packages/cortex/src/cortex/stacks/auto.py

# Build a 4-layer Column stack; each layer mixes A, X, M, S experts
stack = build_cortex_auto_stack(
    d_hidden=256,
    num_layers=4,
    pattern="AXMS",                 # per-layer expert set (can be a list for per-layer patterns)
)

stack = stack.cuda()

# Initialize and run
B, T = 4, 16
x = torch.randn(B, T, 256, device="cuda")
out, state = stack(x)

# Single-step inference
x_step = torch.randn(B, 256, device="cuda")
out_step, state = stack.step(x_step, state)
```

Advanced control:

- Per‑layer patterns: pass a list like `["AXMS", "AM^S", "XXS", "M^"]`; or a single pattern "XXS" repeated with
  `num_layers`.
- Custom symbols: supply `custom_map={"Q": PreUpBlockConfig(cell=mLSTMCellConfig(...))}` and use "Q" in patterns.
- Column implementation: `packages/cortex/src/cortex/blocks/column/column.py`; pattern builder:
  `packages/cortex/src/cortex/blocks/column/auto.py`.

### Global overrides

You can override default configs produced by the token builder by passing `override_global_configs` to the auto stack
builders. Overrides are applied by type across the entire generated config graph (Column → experts → blocks → cells),
merging only the explicitly set fields on your override instance.

Examples:

```python
from cortex.stacks import build_cortex_auto_stack
from cortex.config import XLCellConfig, RouterConfig, PostUpGatedBlockConfig

# 1) Override Transformer‑XL memory length globally (affects X/X^)
stack = build_cortex_auto_stack(
    d_hidden=256,
    num_layers=2,
    pattern="X^X",
    override_global_configs=[XLCellConfig(mem_len=64)],
)

# 2) Change Column router defaults (e.g., key dim and temperature)
stack = build_cortex_auto_stack(
    d_hidden=256,
    num_layers=2,
    pattern="AXMS",
    override_global_configs=[RouterConfig(d_key=128, temperature=0.7)],
)

# 3) Tweak block‑level projection width for gated post‑up (used by token X)
stack = build_cortex_auto_stack(
    d_hidden=256,
    num_layers=2,
    pattern="XX",
    override_global_configs=[PostUpGatedBlockConfig(proj_factor=2.0)],
)
```

Notes:

- Type‑based: every instance matching the override type is updated.
- Explicit‑fields only: only fields you set on the override are merged; others keep their original values (e.g., `X^`
  still enables `use_axon_qkv=True`).
- `cell.hidden_size` is always inferred from the enclosing block/stack and cannot be overridden.
- Per‑layer targeting is not supported by this API; provide a custom pattern/map if you need per‑layer differences.

## Supported Components

### Memory Cells

Core computational units implementing recurrent logic. All cells follow batch-first convention: `[B, T, H]` for
sequences, `[B, H]` for single-step.

| Cell           | Description                                                                                                 | Triton Accelerated        | CUDA Accelerated         |
| -------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------ |
| `LSTMCell`     | Stateless LSTM wrapper with TensorDict state (`h`, `c`); step and sequence parity; optional resets.         | Yes                       | No                       |
| `mLSTMCell`    | Matrix-LSTM with per-head state, chunkwise closed-form updates, and optional causal Conv1D pre-activation.  | Yes                       | No                       |
| `sLSTMCell`    | Structured LSTM with per-head gating, stabilized accumulators (`c`, `n`, `m`), and optional causal Conv1D.  | Yes                       | No                       |
| `CausalConv1d` | Depthwise causal Conv1D cell (ring-buffer state); supports optional channel-mixing mode.                    | Yes (channel-mixing only) | No                       |
| `AxonCell`     | Streaming RTU with diagonal input weights (per-channel local recurrence, 2H→H→out_dim projection).          | Yes                       | Yes (seq‑allin, short‑T) |
| `XLCell`       | Transformer‑XL style multi‑head attention with rolling memory; optional AxonLayer‑backed Q/K/V projections. | No                        | No                       |
| `AGaLiTeCell`  | AGaLiTe attention                                                                                           | No                        | Yes (fused discount sum) |

**Notes:**

- Triton kernels are selected automatically on CUDA when constraints are met; otherwise PyTorch fallback is used
- `CausalConv1d` uses Triton only in channel-mixing mode (groups=1) with per-timestep resets
- Resets (episode boundaries) are optional and broadcast-safe: `[B, T]` for sequences, `[B]` for steps

### Blocks

Wrappers around cells that handle projections, normalization, and information flow.

| Block              | Description                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `PassThroughBlock` | Applies the nested cell directly at `d_hidden` with residual; no projections.                                                                    |
| `PreUpBlock`       | Pre-upsamples to `d_inner = int(proj_factor * d_hidden)`, runs the cell at `d_inner`, gates and projects back to `d_hidden`, then adds residual. |
| `PostUpBlock`      | Runs the cell at `d_hidden`, then applies a gated feed-forward projection up and back down before residual. Useful for deep stacks.              |
| `PostUpGatedBlock` | Like `PostUpBlock` but with GRU‑style gating (GTrXL‑inspired) for both sublayers (cell and FFN) to stabilize deep training.                      |
| `AdapterBlock`     | Wraps another block with a trainable residual adapter (identity at init). Lets you insert capacity without changing behavior at t=0.             |

#### Hidden Size Inference in Blocks

**Important**: Some blocks control the working dimension of their nested cell and will **override** the cell's
`hidden_size` during stack construction, regardless of what value you provide:

- `PreUpBlock`: Sets `cell.hidden_size = int(proj_factor * d_hidden)`
- `PostUpBlock`: Sets `cell.hidden_size = d_hidden`
- `PassThroughBlock`: Sets `cell.hidden_size = d_hidden`

**Best Practice**: Set `hidden_size = None` in the cell config to make the override explicit and avoid confusion. The
builder will infer and set the correct value automatically.

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

This override happens only when building via `CortexStackConfig`/`build_cortex`. If you instantiate blocks and cells
directly (without the stack builder), you must provide concrete sizes that satisfy these relationships manually.

## MoE using Column

Column is a mixture-of-experts block that runs multiple expert blocks in parallel and routes their deltas through a
router. A global prior gate (with optional per-token refinement) selects the top‑k experts per token, an E‑axis mixer
stabilizes cross-expert interactions, and an outer ReZero aligns residuals.

Build a Column from a compact token pattern:

```python
from cortex.blocks.column.auto import build_column_auto_config
from cortex.config import RouterConfig
import cortex.tokens  # ensure built‑in tokens are registered via decorators

col_cfg = build_column_auto_config(
    d_hidden=256,
    pattern="AXMS^",                 # A|X|M|S experts; ^ enables axonified variant when supported
    router=RouterConfig(top_k=2, temperature=0.7),
)
```

Customize experts per symbol (caret requests axonify known cells X/M/S when only base is provided):

```python
from cortex.config import PostUpGatedBlockConfig, XLCellConfig

custom = {"X": PostUpGatedBlockConfig(cell=XLCellConfig(mem_len=128))}
col_cfg2 = build_column_auto_config(d_hidden=256, pattern="X^", custom_map=custom)
```

Built‑in expert tokens:

- `A` = Axon (PostUp)
- `Ag` = AGaLiTe (PostUpGated)
- `X` = Transformer‑XL (PostUpGated), `X^` axonified QKV
- `M` = mLSTM (PreUp), `M^` axonified gates+QKV
- `S` = sLSTM (PostUp), `S^` axonified gates
- `L` = LSTM (PassThrough)
- `C` = CausalConv1d (PassThrough)

### Compact Forward Pass (per token t)

$$
\begin{aligned}
u_t &:= \mathrm{RMSNorm}(x_t) \\
y_{t,i} &= \mathrm{Block}_i(u_t) \\
\Delta_{t,i} &= y_{t,i} - u_t \\
\tilde{\Delta}_{t,i} &= \Delta_{t,i} + \mathrm{Mixer}(\Delta)_{t,i}
\quad\text{(cross-attention over experts }E\text{)} \\
\alpha_t &= \mathrm{softmax}\!\big(\log \mathrm{softmax}(z_g) + \lambda\, \hat{p}_t\big)
\quad (\alpha_{t,i}\ge 0,\ \sum_i \alpha_{t,i}=1) \\
r_t &= \sum_i \alpha_{t,i}\,\tilde{\Delta}_{t,i} + (u_t - x_t)
\quad\text{(align from normalized space }u_t\text{ back to }x_t\text{)} \\
y_{\mathrm{total}}(t) &= x_t + r_t \\
\mathrm{out}_t &= y_{\mathrm{total}}(t) + \alpha_{\mathrm{col}} \cdot \rho(r_t)\, .
\end{aligned}
$$

## Advanced Setup

Compose stacks manually by specifying columns, blocks and cells directly. This mirrors what the DSL expands to and is
useful for full control or experimentation.

```python
import torch
from cortex import CortexStackConfig, build_cortex
from cortex.config import LSTMCellConfig, PreUpBlockConfig, PassThroughBlockConfig

config = CortexStackConfig(
    d_hidden=256,
    blocks=[
        PreUpBlockConfig(cell=LSTMCellConfig(hidden_size=None, num_layers=2), proj_factor=2.0),
        PassThroughBlockConfig(cell=LSTMCellConfig(hidden_size=256, num_layers=1)),
    ],
    post_norm=True,
)

stack = build_cortex(config)
B, T = 4, 16
state = stack.init_state(batch=B, device="cuda", dtype=torch.float32)
x = torch.randn(B, T, 256, device="cuda")
out, state = stack(x, state)
```

### Register tokens and build via `build_cortex`

You can register custom expert tokens and still build stacks manually with `build_cortex` by first creating Column
configs from token patterns.

```python
# 1) Define and register custom tokens
from cortex.registry import register_token
from cortex.config import PostUpGatedBlockConfig, PreUpBlockConfig, XLCellConfig, mLSTMCellConfig

@register_token("Y")
def build_Y():
    return PostUpGatedBlockConfig(cell=XLCellConfig(mem_len=64))

@register_token("Y^")
def build_Y_axon():
    d = XLCellConfig().model_dump()
    d["use_axon_qkv"] = True
    return PostUpGatedBlockConfig(cell=XLCellConfig(**d))

@register_token("Q")
def build_Q():
    return PreUpBlockConfig(cell=mLSTMCellConfig())

# 2) Build Columns from a pattern and compose a stack config
import cortex.tokens  # ensure built-ins are registered
from cortex.blocks.column.auto import build_column_auto_config
from cortex import CortexStackConfig, build_cortex

col1 = build_column_auto_config(d_hidden=256, pattern="AY^Q")
col2 = build_column_auto_config(d_hidden=256, pattern="AXMS^")
cfg = CortexStackConfig(d_hidden=256, blocks=[col1, col2], post_norm=True)
stack = build_cortex(cfg)

# Alternatively, use custom_map at runtime (no decorators)
custom_map = {
    # New custom symbols
    "Y": PostUpGatedBlockConfig(cell=XLCellConfig(mem_len=64)),
    "Q": PreUpBlockConfig(cell=mLSTMCellConfig()),
    # Override built‑in X; requesting "X^" will axonify this config automatically
    "X": PostUpGatedBlockConfig(cell=XLCellConfig(mem_len=32)),
}

# Use separators for custom symbols; caret works for built‑ins (X/M/S)
col3 = build_column_auto_config(d_hidden=256, pattern="Y X^ Q", custom_map=custom_map)
cfg2 = CortexStackConfig(d_hidden=256, blocks=[col3], post_norm=True)
stack2 = build_cortex(cfg2)
```

---

## AxonLayer: A Generalized Linear Operator with Stateful Dynamics

`AxonLayer` provides a stateful generalization of the standard `nn.Linear(in_features → out_features)` operator. Instead
of performing a purely affine transformation, it integrates a lightweight **recurrent dynamic** through an internal
[`AxonsCell`](../cells/axons.py), which maintains structured state evolution across timesteps. Each forward invocation
reads and **mutates a caller-provided parent `TensorDict` state in place**, enabling temporally coherent computation
without requiring explicit recurrence or sequence unrolling in the computation graph.

At a high level, `AxonLayer` functions as a **streaming linear projection** — a linear operator augmented with local
recurrence and compact temporal traces. This design yields temporal expressivity comparable to recurrent layers while
retaining the simplicity and efficiency of a feed-forward linear transformation. Within the Cortex framework,
`AxonLayer` integrates seamlessly with the `MemoryCell` and `TensorDict` ecosystem, making it composable and
state-consistent with other core primitives.

```python
from tensordict import TensorDict
from cortex.cells.base import MemoryCell
from cortex.cells.core import AxonLayer, update_parent_state
from cortex.config import AxonConfig

class MyCell(MemoryCell):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size=hidden_size)
        ax_cfg = AxonConfig(hidden_size=hidden_size, out_dim=hidden_size)
        self.ax = AxonLayer(hidden_size, hidden_size, cfg=ax_cfg, name="proj", group="mycell")
        # ... additional layer definitions ...

    def forward(self, x, state):
        if state is None:
            state = TensorDict({}, batch_size=[x.shape[0]])

        ax = self.ax(x, state=state)  # mutates state in place
        # ... custom logic using ax ...
        out = ax  # placeholder

        next_state = TensorDict({
            # "c": ..., "n": ..., "m": ...  # cell-specific keys
        }, batch_size=[x.shape[0]])
        update_parent_state(next_state, state)
        return out, next_state
```

### Internal Structure

Internally, `AxonLayer` encapsulates an **`AxonCell`**, which serves as the computational core responsible for local
recurrence and gradient-preserving temporal traces. `AxonCell` generalizes the behavior of a linear layer through
several defining mechanisms:

- **Linear→Recurrent Transition:** Each channel is augmented with a _diagonal Recurrent Transition Unit (RTU)_ update,
  producing a doubled activation (`2H`) that is projected back to the target dimension (`H`). This turns a standard
  `Linear(H → H)` operation into a _locally recurrent_ one with minimal parameter overhead.

- **Optional SRHT Feature Mixer:** Prior to the main kernel, `AxonCell` may apply a **Subsampled Randomized Hadamard
  Transform (SRHT)** to orthogonally mix feature channels. This improves conditioning, allowing per-channel diagonal
  updates to behave more like dense transformations. It includes both a CUDA fast path (power-of-two `H`) and a PyTorch
  fallback (arbitrary `H`), controlled via `AxonConfig.use_srht` and `AxonConfig.srht_permute`. The transform is
  normalized by `1/√H` to preserve scale and gradient norm.

- **Streaming Traces:** Instead of storing full activation histories, `AxonCell` maintains compact **eligibility
  traces** across subsequences. A boundary correction at chunk heads preserves cross-chunk credit assignment,
  effectively extending learning beyond TBPTT truncation limits.

These mechanisms collectively allow `AxonLayer` to act as a _locally recurrent linear primitive_, efficiently
propagating gradient information across time without explicit recurrent loops or large memory footprints.

### Rationale

- **State-Augmented Linear Transformation:** `AxonLayer` transforms the conventional linear projection into a
  state-aware operator capable of encoding short- and medium-term temporal dependencies.

- **Gradient Flow Beyond TBPTT:** Through compact trace preservation, it allows credit signals to propagate beyond
  truncated backpropagation windows — a key step toward continuous, long-horizon learning.

- **Architectural Role:** As the canonical linear primitive for upcoming Cortex architectures, `AxonLayer` embeds
  temporal inductive bias directly into the model’s feed-forward backbone, replacing static linear projections with
  dynamically evolving connections.

---

### AxonLayer Integration Across Cells

The table below tracks AxonLayer replacements for linear-like projections in key cells. AxonLayer is a stateful
alternative to `nn.Linear` that wraps `AxonCell` and updates per-layer state inside a parent `TensorDict`.

| Cell                                   | Components Replaced                                                                     | Flags                            | State Group/Keys                                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| sLSTM (`cortex.cells.slstm.sLSTMCell`) | Fused gate projections: `if_fused` (i,f) from conv path; `zo_fused` (z,o) from seq path | `use_axon_layer`                 | group=`slstm`, keys=`if_fused`, `zo_fused` (compat keys like `{igate,fgate,zgate,ogate}_h{i}` may also appear) |
| mLSTM (`cortex.cells.mlstm.mLSTMCell`) | Input/forget gates; optional QKV path                                                   | `use_axon_layer`, `use_axon_qkv` | group=`mlstm` keys=`igate`,`fgate`; optional group=`mlstm_qkv` keys=`qk`,`v`                                   |
| XL (`cortex.cells.xl.XLCell`)          | Q/K/V projections (optional)                                                            | `use_axon_qkv`                   | group=`xl_qkv`, keys=`q`, `k`, `v`                                                                             |

Note: AxonLayer usage is opt-in per cell via its config (e.g., `use_axon_layer`, `use_axon_qkv`). The layer mutates the
provided parent TensorDict state in place. When building stacks with the auto-pattern DSL, you can enable these Axon
augmentations inline by using the `^` suffix on supported experts (for example, `M^`, `S^`, or `X^`). The suffix routes
through the AxonLayer-enabled variant of that expert without manually toggling the config flags.

## Metta Framework Integration

Metta ships with a ready-to-use component for integrating Cortex stacks with its TensorDict-based pipelines.

### CortexTD Component

The `CortexTD` component (in `agent/src/metta/agent/components/cortex.py`) wraps a `CortexStack` and provides stateful
memory across rollout and training.

**Recommended pattern (auto stack):** Use the mixed Axon/mLSTM/sLSTM builder and enable AxonLayers.

```python
from cortex.stacks import build_cortex_auto_stack
from metta.agent.components.cortex import CortexTD, CortexTDConfig

# 1) Build a Cortex stack
stack = build_cortex_auto_stack(
    d_hidden=256,
    num_layers=3,
    post_norm=True,
    use_axonlayers=True,
)

# 2) Wrap it as a Metta component
component = CortexTD(CortexTDConfig(
    stack=stack,
    in_key="latent",
    out_key="recurrent_out",
    d_hidden=256,              # stack external size
    out_features=256,          # identity projection when equal to d_hidden
    key_prefix="cortex_state",
    store_dtype="fp32",       # or "bf16"
))
```

See also the reference wiring in:

- `agent/src/metta/agent/components/cortex.py`
- `agent/src/metta/agent/policies/cortex.py`

**TensorDict expectations:**

- `bptt`: int Tensor (shape [1]); 1 for rollout (step mode), >1 for training (sequence mode)
- `batch`: int Tensor (shape [1]); batch size B
- `training_env_ids`: Long Tensor identifying environments (B or [B, T])
- Optional resets via `dones`/`truncateds` booleans (B or [B, T])

`CortexTD` maintains separate caches for rollout and training, supports checkpointing via `get_memory()`/`set_memory()`,
and applies resets automatically.

## Evaluate Quickly

Cortex includes lightweight synthetic tasks for sanity-checking stacks and comparing recipes. These evaluations verify
wiring, state handling, and step/sequence parity.

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

Cortex automatically selects between Triton (GPU-accelerated) and PyTorch backends based on device, dtype, and cell
constraints. You can override this behavior with environment variables:

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

- Triton kernels are used automatically on CUDA for supported cells (`LSTMCell`, `mLSTMCell`, `sLSTMCell`,
  `CausalConv1d`)
- Some cells have additional constraints (e.g., LSTM requires power-of-two `hidden_size`)
- Falls back to PyTorch when constraints aren't met

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

You can define stacks from compact token patterns using the auto builders. Each pattern expands to a Column of
predefined or custom “expert” blocks. See Quick Start for the list of built‑in tokens and caret `^` semantics.

1. Register custom tokens (optional)

Register new symbols with a decorator. You can also register caret variants explicitly by using the token with a `^`.

```python
# packages/your_pkg/my_tokens.py
from cortex.registry import register_token
from cortex.config import PostUpGatedBlockConfig, XLCellConfig

@register_token("Y")
def build_Y():
    return PostUpGatedBlockConfig(cell=XLCellConfig())

@register_token("Y^")
def build_Y_axon():
    d = XLCellConfig().model_dump()
    d["use_axon_qkv"] = True
    return PostUpGatedBlockConfig(cell=XLCellConfig(**d))
```

Make sure your module is imported before building (e.g., `import your_pkg.my_tokens`). Built‑ins are loaded by
`cortex.tokens` automatically.

2. Build from a pattern

Use the auto builders to construct a `CortexStackConfig` or an instantiated stack from token patterns.

```python
from cortex.stacks import build_cortex_auto_config, build_cortex_auto_stack
import cortex.tokens  # ensure built‑ins are registered via decorators

cfg = build_cortex_auto_config(d_hidden=256, num_layers=3, pattern="Y^Y")
stack = build_cortex_auto_stack(d_hidden=256, num_layers=3, pattern=["YY", "Y^", "YYY"])
```

4. Quick check

Instantiate a stack and run a short forward pass as shown in Quick Start.
