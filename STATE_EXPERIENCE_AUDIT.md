# State and Experience Management Audit: ComponentPolicy vs py_agent

## Executive Summary

After comprehensive analysis of the state and experience handling in ComponentPolicy vs py_agent implementations, I've identified critical missing functionality that could explain the performance discrepancy and training collapse.

## 1. Experience Buffer Integration

### What ComponentPolicy Does:
1. **TensorDict Field Management**:
   - Sets `td["bptt"]` and `td["batch"]` in forward() for LSTM context
   - These fields are **critical** for LSTM state management in ComponentPolicy
   - The LSTM component uses these fields to properly reshape hidden states

2. **Experience Buffer Spec**:
   - MettaAgent.get_agent_experience_spec() only returns `env_obs`
   - The trainer adds loss-related fields: rewards, dones, truncateds, actions, act_log_prob, values, returns
   - Experience buffer selects only the fields it needs via `data_td.select(*self.buffer.keys())`

3. **Training Environment ID**:
   - Trainer sets `td["training_env_id_start"]` for each batch
   - This tracks which environment (slice) the data comes from
   - Critical for per-environment LSTM state management

### What py_agent Currently Does:
1. **Missing TensorDict Fields**:
   - Fast.forward() does NOT set `td["bptt"]` and `td["batch"]`
   - These are required by ComponentPolicy's LSTM but missing in py_agent
   - Without these, LSTM state management may be incorrect during training

2. **Incomplete Experience Spec**:
   - py_agent policies inherit the same basic get_agent_experience_spec()
   - This only includes `env_obs`, missing potential policy-specific fields

## 2. LSTM State Management Differences

### ComponentPolicy's LSTM (lib/lstm.py):
```python
def _forward(self, td: TensorDict):
    TT = td["bptt"][0]  # REQUIRES this field
    B = td["batch"][0]   # REQUIRES this field
    
    # Uses these to reshape hidden states properly
    hidden = rearrange(hidden, "(b t) h -> t b h", b=B, t=TT)
```

### py_agent's LSTM (base.py):
```python
def _manage_lstm_state(self, td, B, TT, device):
    # Calculates B and TT locally from observations shape
    # Does NOT use td["bptt"] or td["batch"]
    # This means it might not align with how the trainer expects it
```

## 3. Critical Missing Functionality in py_agent

### Issue 1: Missing TensorDict Fields During Training
**Problem**: py_agent doesn't set `td["bptt"]` and `td["batch"]` which ComponentPolicy's training loop expects.

**Impact**: 
- During minibatch training, the policy might not know the correct batch structure
- LSTM states might be reshaped incorrectly
- Could cause gradient issues and training instability

### Issue 2: Inconsistent LSTM State Tracking
**Problem**: ComponentPolicy tracks LSTM state per training_env_id_start, but also uses bptt/batch for reshaping.

**Impact**:
- Mismatch between how states are stored (per env_id) and how they're used (with bptt/batch)
- Could lead to state corruption across episodes

### Issue 3: Memory Reset Timing
**Problem**: The trainer calls `policy.reset_memory()` at two critical points:
1. Before each rollout (line 303)
2. Before each minibatch (line 384)

**Impact**:
- py_agent might not be resetting memory correctly
- ComponentPolicy components handle this via components_with_memory list
- py_agent uses a simpler approach that might miss edge cases

## 4. Experience Storage Flow

### During Rollout:
1. Trainer creates td with: env_obs, rewards, dones, truncateds, training_env_id_start
2. Policy forward() should add: actions, act_log_prob, values, full_log_probs
3. Experience.store() selects only the fields defined in experience_spec
4. Buffer stores data in segmented tensors for BPTT

### During Training:
1. Minibatch sampled with prioritized sampling
2. policy_td created with only policy-relevant fields
3. Policy forward() called with stored observations and actions
4. LSTM states should align with the minibatch structure

## 5. Key Differences Found

### ComponentPolicy:
- Sets td["bptt"] and td["batch"] in every forward pass
- LSTM component depends on these fields for proper operation
- Reshapes output TD based on these fields during training
- Has sophisticated component-based memory management

### py_agent (Fast):
- Does NOT set td["bptt"] and td["batch"]
- Calculates batch structure locally from observation shape
- Might not align with trainer's expectations
- Simpler but potentially incomplete memory management

## 6. Recommendations for Fixing py_agent

### Priority 1: Add TensorDict Fields
```python
def forward(self, td: TensorDict, state=None, action=None):
    observations = td["env_obs"]
    
    # CRITICAL: Set bptt and batch fields like ComponentPolicy does
    if observations.dim() == 4:  # Training: [B, T, obs_tokens, 3]
        B = observations.shape[0]
        TT = observations.shape[1]
        # Flatten and set fields
        td = td.reshape(B * TT)
        td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
        td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
    else:  # Inference: [B, obs_tokens, 3]
        B = observations.shape[0]
        td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
        td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))
```

### Priority 2: Align LSTM State Management
- Ensure LSTM states are managed consistently with bptt/batch structure
- Verify states are reset at the correct times
- Check that per-environment tracking aligns with minibatch structure

### Priority 3: Verify Experience Buffer Integration
- Ensure all required fields are present in TD during training
- Check that the policy outputs match what the experience buffer expects
- Verify that memory resets happen at the right times

## 7. Root Cause Hypothesis

The training collapse in py_agent=fast is likely caused by:

1. **Missing td["bptt"] and td["batch"] fields** causing incorrect tensor reshaping during training
2. **LSTM state misalignment** between rollout collection and minibatch training
3. **Incorrect gradient flow** due to tensor shape mismatches

These issues compound over time, leading to:
- Initial learning (when LSTM states are fresh)
- Plateau (as state corruption accumulates)
- Collapse (when corrupted states dominate training)

## Next Steps

1. Implement the missing TensorDict fields in all py_agent policies
2. Verify LSTM state management aligns with ComponentPolicy
3. Test on GPU to confirm performance parity
4. Add comprehensive logging to track tensor shapes and LSTM states during training