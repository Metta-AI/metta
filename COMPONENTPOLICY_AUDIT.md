# ComponentPolicy vs Fast.py Deep Audit

## Critical Findings

### 1. LSTM Initialization Differences

**ComponentPolicy (lstm.py:75-78):**
```python
for name, param in net.named_parameters():
    if "bias" in name:
        nn.init.constant_(param, 1)  # Joseph originally had this as 0
    elif "weight" in name:
        nn.init.orthogonal_(param, 1)  # torch's default is uniform
```

**Fast.py (lines 44-49):**
```python
for name, param in self.lstm.named_parameters():
    if "bias" in name:
        nn.init.constant_(param, 1)  # Match YAML agent initialization
    elif "weight" in name:
        nn.init.orthogonal_(param, 1)  # Orthogonal initialization
```

✅ **Status**: Matching - both use bias=1, orthogonal weight init with gain=1

### 2. LSTM State Management

**ComponentPolicy LSTM (lstm.py:94-117):**
- Uses dictionary storage keyed by `training_env_id_start`
- Resets hidden state on done/truncated episodes
- Detaches hidden states to prevent gradient leakage
- Uses `@torch._dynamo.disable` decorator to avoid graph breaks

**Fast.py:**
- Simple state management without env_id tracking
- No automatic reset on done/truncated
- No detach() calls on stored states

🔴 **ISSUE**: Fast.py missing critical LSTM state management features

### 3. Action Head Implementation

**ComponentPolicy (MettaActorSingleHead):**
```python
# Lines 149-153 in actor.py
query = torch.einsum("n h, k h e -> n k e", hidden_reshaped, self.W)
query = self._tanh(query)  # Tanh BEFORE scores
scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)
biased_scores = scores + self.bias
```

**Fast.py (lines 379-383):**
```python
query = torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W)
query = torch.tanh(query)  # Matches ComponentPolicy
scores = torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped)
biased_scores = scores + self.actor_bias
```

✅ **Status**: Matching implementation

### 4. Observation Processing

**ComponentPolicy (ObsTokenToBoxShaper):**
- Handles invalid tokens with coords_byte == 0xFF
- Validates attribute indices < num_layers
- Uses scatter_ operation for efficient tensor population
- Warns about out-of-range observation channels

**Fast.py (encode_observations):**
- Sets 255 values to 0 (line 324)
- Validates coordinates and attributes
- Direct indexing instead of scatter_

⚠️ **DIFFERENCE**: Different approaches to handling invalid tokens

### 5. Tensor Reshaping in Forward Pass

**ComponentPolicy (forward, lines 125-134):**
```python
if td.batch_dims > 1:
    B = td.batch_size[0]
    TT = td.batch_size[1]
    td = td.reshape(td.batch_size.numel())  # flatten to BT
    td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
    td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
```

**Fast.py (forward, lines 93-94):**
```python
B = observations.shape[0]
TT = 1 if observations.dim() == 3 else observations.shape[1]
```

🔴 **ISSUE**: Fast.py doesn't set "bptt" and "batch" in TensorDict, which LSTM component expects

### 6. Weight Clipping Implementation

**ComponentPolicy:**
- Calls `_apply_to_components("clip_weights")` which delegates to each component
- Each component can have its own clipping logic

**Fast.py:**
- Relies on MettaAgent's default implementation
- Single uniform clipping across all parameters

⚠️ **DIFFERENCE**: ComponentPolicy allows per-component clipping strategies

### 7. Action Embedding Initialization

**ComponentPolicy (via ActionEmbedding component):**
- Orthogonal initialization then scaled to max 0.1
- Dynamic activation based on action names

**Fast.py (lines 269-275):**
```python
nn.init.orthogonal_(self.action_embeddings.weight)
with torch.no_grad():
    max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
    self.action_embeddings.weight.mul_(0.1 / max_abs_value)
```

✅ **Status**: Matching implementation

### 8. Critical Missing Features in Fast.py

1. **LSTM Memory Management**:
   - No `has_memory()`, `get_memory()`, `reset_memory()` methods
   - No per-environment hidden state tracking
   - No automatic reset on episode boundaries

2. **TensorDict Integration**:
   - Doesn't set required "bptt" and "batch" keys
   - LSTM component expects these for proper reshaping

3. **Gradient Detachment**:
   - LSTM states not detached, causing potential gradient leakage
   - Could lead to exploding gradients during BPTT

4. **Dynamic Disable**:
   - No `@torch._dynamo.disable` on LSTM forward
   - May cause compilation issues with torch.compile

## Recommendations for Fixing Fast.py

1. **High Priority**:
   - Add proper LSTM state management with detach()
   - Set "bptt" and "batch" in TensorDict
   - Implement memory management methods

2. **Medium Priority**:
   - Add done/truncated episode handling
   - Match observation invalid token handling exactly
   - Add per-environment state tracking

3. **Low Priority**:
   - Add @torch._dynamo.disable decorator
   - Implement component-specific weight clipping

## Summary

The most critical issues are in LSTM state management. ComponentPolicy's LSTM:
1. Detaches hidden states to prevent gradient accumulation
2. Resets states on episode boundaries
3. Tracks states per environment
4. Sets required TensorDict keys for downstream components

Fast.py is missing these crucial features, which could explain training instability and collapse.