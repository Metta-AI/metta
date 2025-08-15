# Fix Summary: py_agent=fast Training Collapse

## Problem
The `py_agent=fast` configuration was plateauing in reward and then collapsing during training, while the `agent=fast` (YAML-based) configuration trained successfully.

## Root Cause
A shape mismatch bug in the `Fast` class (used by `py_agent=fast`) when handling training mode with BPTT (Backpropagation Through Time).

### Specific Issue
In `/Users/relh/Code/workspace/metta/agent/src/metta/agent/pytorch/fast.py`, line 144:
- **Bug**: Used wrong variable `T` instead of `TT` when reshaping `full_log_probs`
- `T` came from the action tensor shape (line 129)
- `TT` came from the observation tensor shape (line 89)
- These can differ during BPTT sequences

### Secondary Issue
Line 145 was not preserving the value tensor's last dimension when reshaping.

## Fix Applied

```python
# Line 144 - Fixed variable name
td["full_log_probs"] = action_log_probs.view(B, TT, -1)  # Changed T to TT

# Line 145 - Preserved value dimension
td["value"] = value.view(B, TT, -1)  # Added -1 to preserve last dimension
```

## Why This Caused Training Collapse

1. **Shape Mismatch**: The incorrect reshaping caused tensor shape mismatches during the backward pass
2. **Gradient Corruption**: Shape mismatches led to incorrect gradient calculations
3. **Training Instability**: Corrupted gradients caused the policy to degrade over time
4. **Reward Collapse**: The degraded policy produced increasingly poor actions, leading to reward collapse

## Verification

The fix ensures that:
1. All tensors are reshaped consistently using the correct time dimension (`TT`)
2. The value tensor maintains its proper shape `[B, TT, 1]`
3. The `py_agent=fast` forward pass now matches the behavior of the YAML `agent=fast`

## Testing
- Created test scripts to verify the forward pass works correctly with BPTT
- Confirmed that tensor shapes match expected dimensions
- Both agent configurations now produce consistent outputs