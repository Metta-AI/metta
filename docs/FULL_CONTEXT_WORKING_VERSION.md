# Full-Context Transformer - Working Version Summary

## Version Identification
- **Commit**: 7bbe069b (richard-transformer branch)
- **Date**: August 16, 2025
- **Model Parameters**: 2,696,066
- **Training Speed**: ~2.4 ksps on CPU

## Critical Fixes That Made It Work

### 1. Observation Encoding (Most Important!)
**Problem**: Was using simple flattened MLP encoder
**Solution**: Implemented scatter-based token placement with CNN processing
```python
# Extract token coordinates and scatter into spatial grid
coords_byte = token_observations[..., 0].to(torch.uint8)
x_coords = ((coords_byte >> 4) & 0x0F).long()
y_coords = (coords_byte & 0x0F).long()
# Scatter into spatial grid then process with CNN
```

### 2. Action Decoding 
**Problem**: Simple linear heads were too weak
**Solution**: Bilinear interaction with action embeddings
```python
# Bilinear interaction for powerful action selection
query = torch.einsum("n h, k h e -> n k e", actor_features, self.actor_W)
scores = torch.einsum("n k e, n e -> n k", query, action_embeds)
```

### 3. GTrXL Architecture
- **Post-normalization**: LayerNorm → Attention/FFN → ReLU → GRU Gate
- **Orthogonal initialization**: All weights use gain=√2
- **GRU gate bias**: Set to 2.0 for identity mapping at initialization
- **ReLU activations**: Consistent throughout (not GELU)

## Configuration to Use

### Command Line (Direct Instantiation)
```bash
uv run ./tools/train.py py_agent=full_context \
  trainer.total_timesteps=1000000 \
  trainer.num_workers=4 \
  wandb=on
```

### Key Hyperparameters
- `hidden_size`: 128 (matches Fast for fair comparison)
- `n_heads`: 8
- `n_layers`: 6
- `d_ff`: 512 (reduced for efficiency)
- `use_gating`: true (GTrXL-style)
- `use_causal_mask`: true

## Performance Characteristics

### Compared to Fast Agent (LSTM baseline)
| Metric | Fast (LSTM) | Full-Context (Transformer) |
|--------|-------------|---------------------------|
| Parameters | 574,978 | 2,696,066 |
| Speed (CPU) | ~3.6 ksps | ~2.4 ksps |
| Memory per agent | O(hidden_size) | O(sequence_length × hidden_size) |
| Context window | Limited by LSTM | Full BPTT trajectory |

### When This Architecture Excels
- Tasks requiring long-term memory (>10 timesteps)
- Complex multi-agent coordination
- Environments with sparse rewards
- Situations where full trajectory context matters

## Files in This Version

### Core Implementation
- `/agent/src/metta/agent/pytorch/full_context.py` - Main agent class
- `/agent/src/metta/agent/modules/full_context_transformer.py` - Transformer with GTrXL
- `/agent/src/metta/agent/modules/transformer_wrapper.py` - BPTT sequence handling

### Critical Components
1. **Policy.encode_observations()** - Scatter-based token → CNN
2. **Policy.decode_actions()** - Bilinear actor with embeddings  
3. **FullContextTransformer** - GTrXL-stabilized transformer
4. **TransformerWrapper** - Handles BPTT sequences

## Debugging Performance

If performance degrades, check:
1. **Observation encoding** - Must use scatter-based, not simple flatten
2. **Action embeddings** - Must be initialized with orthogonal + 0.1 scaling
3. **Normalization** - Must be post-norm (norm → transform → gate)
4. **GRU bias** - Must be 2.0 for identity initialization

## WandB Runs Reference
- `relh.transformer.816.6` - Previous version (possibly missing fixes)
- `relh.transformer.816.7` - Current working version
- Performance difference likely due to observation encoding fix

## Next Optimization Opportunities
1. Implement Flash Attention for faster GPU training
2. Add linear attention variant for O(T) complexity
3. Experiment with sparse attention patterns
4. Try curriculum learning on sequence length