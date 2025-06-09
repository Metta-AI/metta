# Exponential Memory Traces Integration

This document explains how the exponential memory mechanism from "Partially Observable Reinforcement Learning with Memory Traces" (Eberhard et al., ICML 2025) has been integrated into the MettaAgent architecture.

## Overview

The paper demonstrates that exponential memory traces significantly outperform sliding window approaches in partially observable environments. In their T-Maze experiments, memory traces achieved ~80% success rate while window-based and memoryless approaches failed to learn.

## Key Concepts

### Exponential Memory Update Formula
```
trace_new = (1 - λ) * φ + λ * trace_old
```

Where:
- `φ` is the current observation embedding
- `λ` is the decay parameter (0.0 = no memory, 0.985 = long-term memory)
- Multiple traces with different λ values capture information at different temporal scales

### Benefits vs Sliding Windows
- **Memory efficiency**: Exponential traces compress infinite history into fixed-size vectors
- **Temporal hierarchy**: Different λ values capture short-term vs long-term patterns
- **Performance**: 80% success in T-Maze64 vs 0% for window approaches

## Integration Architecture

### 1. Policy State Extension
Extended `PolicyState` to include memory traces alongside LSTM states:

```python
class PolicyState(TensorClass):
    lstm_h: Optional[torch.Tensor] = None
    lstm_c: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
    # New: Memory traces
    memory_traces: Optional[torch.Tensor] = None  # (num_agents, num_traces, trace_dim)
    trace_weights: Optional[torch.Tensor] = None  # For future extensions
```

### 2. Memory Trace Components

#### ExponentialMemoryTraces
- **Location**: `metta/agent/lib/memory_traces.py`
- **Purpose**: Implements the core exponential memory mechanism
- **Features**:
  - Multiple lambda values for multi-scale memory
  - Efficient tensor operations for batch processing
  - Configurable trace dimensions
  - Built-in statistics computation for monitoring

#### MemoryTraceProcessor
- **Purpose**: Integrates memory traces with existing network flow
- **Integration modes**:
  - `concat`: Concatenate base features with memory traces
  - `add`: Add projected memory to base features
  - `attention`: Use attention mechanism to combine features

### 3. Agent Configuration

#### Memory-Enhanced Agent Config
File: `configs/agent/simple_with_memory.yaml`

Key additions to the component pipeline:
```yaml
# After encoded observations
memory_traces:
  _target_: metta.agent.lib.memory_traces.ExponentialMemoryTraces
  sources:
    - name: encoded_obs
  trace_dim: 128
  lambda_values: [0.0, 0.985]

# Integration with main pipeline
obs_with_memory:
  _target_: metta.agent.lib.memory_traces.MemoryTraceProcessor
  sources:
    - name: encoded_obs
    - name: memory_traces
  integration_mode: concat
```

## Usage Examples

### Basic Configuration
```yaml
memory_traces:
  _target_: metta.agent.lib.memory_traces.ExponentialMemoryTraces
  sources:
    - name: encoded_obs
  trace_dim: 128
  lambda_values: [0.0, 0.985]  # Short + long term memory
```

### Advanced Multi-Scale Memory
```yaml
memory_traces:
  _target_: metta.agent.lib.memory_traces.ExponentialMemoryTraces
  sources:
    - name: encoded_obs
  trace_dim: 256
  lambda_values: [0.0, 0.9, 0.99, 0.999]  # Multiple temporal scales
```

### Integration Options
```yaml
obs_with_memory:
  _target_: metta.agent.lib.memory_traces.MemoryTraceProcessor
  sources:
    - name: encoded_obs
    - name: memory_traces
  integration_mode: attention  # More sophisticated than concat
```

## Key Implementation Details

### 1. Component Architecture Compliance
- Follows existing `LayerBase` pattern
- Uses `_initialize()` and `_forward()` methods
- Integrates with TensorDict system
- Supports component dependency graph

### 2. Memory Management
- Traces stored in TensorDict with persistent keys
- Automatic initialization for new episodes
- Efficient tensor operations for batch processing
- Optional statistics computation for monitoring

### 3. Multi-Lambda Support
- Vectorized computation for multiple decay rates
- Configurable number of traces
- Each trace captures different temporal scales
- Combined via learned network for final output

## Performance Considerations

### Memory Usage
- Traces: `(batch_size, num_traces, trace_dim)`
- Typical: `(1024, 2, 128)` = ~1MB per batch
- Scales linearly with number of traces and dimension

### Computational Cost
- O(1) per timestep (vs O(window_size) for sliding windows)
- Vectorized operations across all traces
- Minimal overhead compared to LSTM computation

## Future Extensions

### 1. Adaptive Lambda Values
- Learn decay rates during training
- Per-trace or per-dimension lambda values
- Dynamic adjustment based on environment

### 2. Attention-Based Memory
- Replace simple combination with transformer
- Cross-attention between traces and observations
- Learned relevance weighting

### 3. Hierarchical Memory
- Multi-level trace hierarchies
- Different resolutions for different time scales
- Conditional computation based on relevance

## Testing and Validation

### Recommended Tests
1. **T-Maze Environment**: Test on corridor lengths 64, 128, 256
2. **Memory Requirements**: Compare vs window-based approaches
3. **Ablation Studies**: Different lambda values and trace dimensions
4. **Integration Modes**: Compare concat vs attention integration

### Expected Results
Based on the paper's findings:
- Memory traces should significantly outperform memoryless agents
- Performance should scale with corridor length better than window approaches
- Multiple lambda values should outperform single lambda

## Migration Guide

### From Simple Agent
1. Copy `simple.yaml` to `simple_with_memory.yaml`
2. Add memory_traces component after encoded_obs
3. Add obs_with_memory processor
4. Update _core_ source to use obs_with_memory
5. Test with your environment

### From Existing Agents
1. Identify observation encoding layer
2. Insert memory components after encoding
3. Update downstream component sources
4. Verify TensorDict keys don't conflict
5. Monitor memory usage and performance

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `memory_traces.py` is in correct location
2. **Shape Mismatches**: Check trace_dim matches integration expectations
3. **Memory Growth**: Monitor trace storage in long episodes
4. **Performance**: Compare with/without memory traces

### Debug Tools
- Use `compute_trace_statistics()` for monitoring
- Check TensorDict keys for proper trace storage
- Validate lambda values are in [0,1] range
- Monitor gradient flow through memory components

## References

- Eberhard et al. "Partially Observable Reinforcement Learning with Memory Traces" ICML 2025
- Original implementation: https://github.com/onnoeberhard/memory-traces
- T-Maze environment for testing POMDP capabilities
