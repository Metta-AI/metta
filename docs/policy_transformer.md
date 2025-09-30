# Transformer-XL Architecture Reference

> **Note:** The current `TransformerPolicy` configured with `variant="trxl"`
> mirrors the original Transformer-XL implementation from [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)
> and no longer applies the additional GRU-style gating that earlier revisions used.
> Historical notes on GTrXL remain below for context when comparing design choices.

> **Variants (September 2025):**
> - `TransformerPolicyConfig(variant="gtrxl")` — gated Transformer without memory
> - `TransformerPolicyConfig(variant="trxl")` — vanilla Transformer-XL with layer memory
> - `TransformerPolicyConfig(variant="trxl_nvidia")` — NVIDIA's optimized Transformer-XL core

### Variant overview (September 2025)
- **variant="gtrxl"** retains the GRU-style gating and pre-layernorm identity-map stabilization introduced for reinforcement learning stability while running without transformer-XL memory.
- **variant="trxl"** mirrors the original Transformer-XL decoder with layer-wise memory caching and relative positional attention but no gating modifications.
- **variant="trxl_nvidia"** keeps NVIDIA's optimized Transformer-XL core (including custom bias handling, clamp length and memory window) while sharing the common CNN/actor heads with the other variants.

### Runtime toggles (September 2025)
- `TransformerBackboneConfig.use_gating`: enable/disable GRU-style gating (GTrXL).
- `ext_len`: retain an additional memory tail beyond the sliding window (TRXL/NVIDIA).
- `attn_dropout`: independent attention-dropout rate alongside `dropout`.
- `activation_checkpoint` / `use_flash_checkpoint`: wrap attention blocks with PyTorch or FlashAttention checkpointing to trade compute for memory.
- `allow_tf32`: opt into TF32 matmuls on supported GPUs. We restore the prior flag on exit.
- `use_fused_layernorm`: leverage Apex fused layer norm when installed.
- `MemorySchedulerConfig`: optionally shrink/grow the transformer memory length across milestones during training.

## Paper Summary
**Title**: "Stabilizing Transformers for Reinforcement Learning"  
**Authors**: Emilio Parisotto et al. (2020)  
**arXiv**: https://arxiv.org/abs/1910.06764  

## Core Problem Addressed
Standard transformers are difficult to optimize in reinforcement learning settings due to:
- Gradient instability during online learning
- Poor performance compared to LSTMs in sequential decision-making
- Difficulty handling long-term dependencies in RL environments

## Key Architectural Modifications

### 1. Identity Map Reordering (Pre-Norm Architecture)
**Standard Transformer**: `output = LayerNorm(input + SubLayer(input))`  
**GTrXL**: `output = input + SubLayer(LayerNorm(input))`

- Layer normalization is applied **before** attention and feed-forward layers
- Creates an identity mapping from first layer input to final layer output
- Enables more stable gradient flow through the network
- Allows the network to more easily learn identity mappings when needed

### 2. Gating Mechanisms
**Replaces standard residual connections with GRU-style gates**

```python
# Instead of: output = input + sublayer_output
# GTrXL uses gated connections:
gate = sigmoid(W_g * [input, sublayer_output] + b_g)
output = gate * sublayer_output + (1 - gate) * input
```

**Benefits**:
- Controls information flow more precisely than residual connections
- Reduces gradient explosion/vanishing issues
- Allows network to selectively forget or remember information

### 3. Layer Normalization Placement
- Applied only to the **input stream** of submodules
- **NOT** applied to gated connections
- Two separate LayerNorm instances:
  - `layernorm1`: Before attention mechanism  
  - `layernorm2`: Before feed-forward network

## Mathematical Formulation

### GTrXL Layer Forward Pass
```python
# Input: x, memory (optional)
# Step 1: Self-attention with pre-norm
norm1_x = layernorm1(x)
attn_out = self_attention(norm1_x, memory)

# Step 2: Gated residual connection
if gating:
    x = gate1(x, attn_out)  # GRU-style gating
else:
    x = x + attn_out        # Standard residual

# Step 3: Feed-forward with pre-norm  
norm2_x = layernorm2(x)
ff_out = feed_forward(norm2_x)

# Step 4: Second gated residual connection
if gating:
    output = gate2(x, ff_out)
else:
    output = x + ff_out
```

### Gating Implementation Details
From OpenDILab implementation:
```python
class GRUGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_r = nn.Linear(d_model, d_model, bias=False)
        self.u_r = nn.Linear(d_model, d_model, bias=False)
        self.w_z = nn.Linear(d_model, d_model, bias=False)  
        self.u_z = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)
        self.u_g = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, y):
        # x: previous layer output, y: current sublayer output
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        z = torch.sigmoid(self.w_z(y) + self.u_z(x))  
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))
        g = (1 - z) * x + z * h
        return g
```

## Memory Mechanism
- Maintains memory of previous sequence chunks
- Enables handling of sequences longer than model context
- Memory is concatenated with current input for attention computation
- Gradients are stopped through memory to prevent instability

## Key Hyperparameters

### Architecture Parameters
- `d_model`: Hidden dimension (typically 256, 512, or larger)
- `n_heads`: Number of attention heads (typically 8 or 16)
- `n_layers`: Number of transformer blocks (typically 6-12)
- `d_ff`: Feed-forward hidden dimension (typically 4 * d_model)
- `memory_len`: Length of memory to maintain (typically 64-256)

### Training Parameters  
- `gru_gating`: Enable/disable gating mechanism (default: True)
- `bias`: Whether to use bias in linear layers
- `dropout`: Dropout rate (typically 0.0-0.2 for RL)
- `layer_norm_eps`: Layer normalization epsilon (typically 1e-5)

## Performance Characteristics
- **Memory environments**: Surpasses LSTMs on tasks requiring long-term memory
- **DMLab-30**: Achieves state-of-the-art results on multi-task benchmark
- **Stability**: More stable training compared to standard transformers in RL
- **Scalability**: Can handle longer sequences through memory mechanism

## Implementation Checklist

### Core Architecture
- [ ] Pre-normalization (LayerNorm before sublayers)
- [ ] GRU-style gating mechanisms  
- [ ] Proper memory management and gradient stopping
- [ ] Attention masking for causal modeling
- [ ] Positional embeddings compatible with memory

### Specific Details
- [ ] Two separate gating layers per transformer block
- [ ] LayerNorm applied to input stream only (not gated outputs)
- [ ] Memory concatenation in attention computation
- [ ] Gradient stopping through memory connections
- [ ] Proper initialization of gating parameters

### Integration Points
- [ ] Compatible with recurrent memory interface
- [ ] Proper handling of episode boundaries  
- [ ] Sequence length flexibility
- [ ] Device placement consistency (CPU/GPU)

## Common Pitfalls to Avoid
1. **Post-norm vs Pre-norm**: Ensure LayerNorm comes BEFORE sublayers
2. **Gating parameter initialization**: Poor initialization can cause gradient issues
3. **Memory gradient flow**: Must stop gradients through memory
4. **Sequence length handling**: Must handle variable sequence lengths properly
5. **Episode boundary handling**: Memory should be reset between episodes

## References
- Original Paper: https://arxiv.org/abs/1910.06764
- OpenDILab Implementation: https://github.com/opendilab/PPOxFamily/blob/main/chapter5_time/gtrxl.py
- DI-engine Documentation: https://opendilab.github.io/DI-engine/12_policies/gtrxl.html

## Current Metta Implementation Analysis

### What We Have Right ✅
1. **Pre-normalization architecture**: Correctly implemented in TransformerBlock:178-246
2. **GRU-style gating**: Implemented via FusedGRUGating class with proper mathematical formulation
3. **Identity map reordering**: LayerNorm applied before attention and FFN operations
4. **Proper initialization**: Gates initialized to favor identity mapping (bg=2.0)
5. **Fused operations**: QKV projection and gating operations optimized for performance

### What Needs Improvement ⚠️
1. **Memory mechanism**: Currently missing - transformer processes full context but lacks memory
2. **Positional encoding**: Using standard sinusoidal, should consider relative positional encoding
3. **Action attention system**: Current implementation is overly complex vs standard GTrXL
4. **Memory gradient handling**: No gradient stopping through memory (since memory is missing)
5. **Episode boundary handling**: Missing proper reset mechanisms

### Key Differences from Reference Implementations
- **OpenDILab GTrXL**: Has explicit memory management with gradient stopping
- **Our implementation**: Full-context processing without memory mechanism  
- **Paper GTrXL**: Emphasizes memory for handling long sequences beyond context window

## Implementation Status (September 2025)
- [x] variant="gtrxl": Legacy gating stack retained and audited against the stabilized RL reference implementation.
- [x] variant="trxl": Mirrors Transformer-XL decoder with layer memory, relative attention, and pre-norm residual flow.
- [x] variant="trxl_nvidia": Wraps NVIDIA's optimized Transformer-XL core with reference-default hyperparameters and bias handling.
- [ ] Extended regression coverage: schedule long-horizon RL smoke tests that exercise memory resets and large batch inference.

## Variant Feature Matrix
| Feature | variant="gtrxl" | variant="trxl" | variant="trxl_nvidia" |
|---------|--------------------|---------------|-------------------------|
| Residual gating | ✅ GRU-style gates | ❌ | ❌ |
| Pre-layer normalization | ✅ | ✅ | ❌ (matches NVIDIA post-norm) |
| Layer count (default) | 4 | 4 | 8 |
| Hidden size (`d_model`) | 64 | 64 | 256 |
| Feed-forward size | 256 | 256 | 1024 |
| Memory length | 0 (stateless) | 32 | 96 |
| Relative positional bias | Sinusoidal, causal mask | Relative bias + memory | NVIDIA partial-relative bias |
| Dropout | 0.05 | 0.05 | 0.05 |
| Attention dropout | 0.05 | 0.05 | 0.0 |

## Default Hyperparameters (Base Config)
| Parameter | variant="gtrxl" | variant="trxl" | variant="trxl_nvidia" |
|-----------|--------------------|---------------|-------------------------|
| `latent_size` / `hidden_size` | 64 | 64 | 256 |
| `num_layers` | 4 | 4 | 8 |
| `n_heads` | 4 | 4 | 4 |
| `d_ff` / `d_inner` | 256 | 256 | 1024 |
| `max_seq_len` | 256 | 256 | 192 |
| `memory_len` | 0 | 32 | 96 |
| `pre_lnorm` | True | True | False |
| `manual_init` (policy heads) | False | False | True |
| Suggested optimizer LR | 7.5e-4 | 9.0e-4 | 3.0e-4 |

## Behavioural Notes
- variant="gtrxl" retains gating to stabilize on-policy RL gradients as described by Parisotto et al. and omits Transformer-XL memory to match historically successful training runs.
- variant="gtrxl" now exposes a slimmed-down constructor focused on the gated architecture used in our RL agents.
- variant="trxl" reinstates layer-wise memory caching and relative attention from the original Transformer-XL formulation for long context handling while keeping the simplified actor/critic heads used internally.
- variant="trxl_nvidia" uses the NVIDIA Megatron-style Transformer-XL block with post-norm residuals, clamp-free relative positional embeddings, and longer default memory to reflect the `wt103_base` recipe.

## Pending Documentation Work
- Expand the variant overview into a comparison table that includes empirical metrics once new benchmarks finish.
- Add illustrated call-outs that show the memory interface flow for TRXL and TRXLNvidia variants.
