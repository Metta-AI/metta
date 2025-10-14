# ViT Consistent Dropout Policy

This document describes the implementation of the ViT policy architecture with consistent dropout for reinforcement learning.

## Overview

The `ViTConsistentDropoutConfig` policy is a variant of the standard VIT policy that incorporates **consistent dropout** to ensure stable policy gradients during reinforcement learning training. This implementation is based on the research by Hausknecht & Wagener, which demonstrated that using different dropout masks during rollout and gradient computation can lead to biased policy gradients and training instability.

## What is Consistent Dropout?

Consistent dropout is a technique that ensures the **same dropout mask is used during both rollout (action selection) and training (gradient computation)**. This prevents the policy gradient from being biased toward a different policy than the one that was actually evaluated.

### The Problem with Standard Dropout in RL

In supervised learning, dropout works by randomly disabling neurons during training. However, in policy gradient methods like PPO:

1. **During rollout**: The policy selects actions using one dropout mask
2. **During training**: The policy is evaluated with a different dropout mask
3. **Result**: The gradient updates a different policy than the one that generated the actions, leading to biased gradients and potential NaN explosions

### The Solution: Consistent Dropout

Consistent dropout solves this by:

1. **Sampling a mask once during rollout**
2. **Storing the mask in the trajectory data**
3. **Reusing the same mask during gradient computation**

This ensures the policy being optimized is exactly the policy that was evaluated.

## Architecture

The ViT Consistent Dropout policy uses the same architecture as the standard VIT policy, with consistent dropout applied in two key locations:

### 1. ObsPerceiverLatent Encoder

- **Location**: `agent/src/metta/agent/components/obs_enc_consistent_dropout.py`
- **Component**: `ObsPerceiverLatentConsistentDropout`
- **Dropout locations**:
  - After attention output projection in each layer
  - After MLP output in each layer
- **Dropout probability**: 0.2 (configurable)

### 2. Actor MLP

- **Location**: `agent/src/metta/agent/components/misc_consistent_dropout.py`
- **Component**: `MLPConsistentDropout`
- **Dropout locations**:
  - After each hidden layer (before the final output layer)
- **Dropout probability**: 0.2 (configurable)

### Critic Path

The **critic (value function)** uses **standard MLP without consistent dropout**. This is intentional because:

- Value function estimation doesn't require the same mask consistency
- The critic is not directly optimized via policy gradients
- Standard dropout provides sufficient regularization for value estimation

## Implementation Details

### Mask Storage and Reuse

Both components use TorchRL's `ConsistentDropout` module, which handles mask creation and reuse automatically:

```python
# In forward pass
mask_key = (self.config.name, f"attn_dropout_mask_{i}")

# Try to retrieve existing mask from TensorDict
attn_mask = td.get(mask_key, None)

if self.training:
    # If mask exists, it will be reused; if None, a new one is created
    attn_output, attn_mask = self.dropout(attn_output, mask=attn_mask)
    # Store mask back in TensorDict for reuse during training
    td[mask_key] = attn_mask
else:
    # In eval mode, dropout is a no-op
    attn_output = self.dropout(attn_output, mask=attn_mask)
```

**Key points:**
1. **Rollout (first pass)**: `mask=None` → new mask is created and stored in TensorDict
2. **Training (second pass)**: `mask=<existing>` → same mask is retrieved and reused
3. **Automatic handling**: If mask exists in TensorDict, it's reused; otherwise, a new one is created

### Training vs Evaluation

The components respect PyTorch's `train()` / `eval()` modes:

- **Training mode** (`model.train()`): Dropout is active, masks are sampled/reused
- **Evaluation mode** (`model.eval()`): Dropout is disabled, no masks are applied

To enable MC-dropout style uncertainty at evaluation:
```python
model.train()  # Keep dropout active
with torch.no_grad():
    action, _ = model(obs)  # Will sample new masks per forward pass
```

## Usage

### Training with Consistent Dropout

You can train using the consistent dropout policy by using the recipe:

```bash
# Train with the consistent dropout VIT policy
uv run ./tools/run.py train abes.vit_consistent_dropout run=my_experiment

# Or specify explicitly in code
from metta.agent.policies.vit_consistent_dropout import ViTConsistentDropoutConfig

policy = ViTConsistentDropoutConfig(
    _dropout_p=0.2  # Configure dropout probability
)
```

### Configuring Dropout Probability

The dropout probability can be adjusted in the policy configuration:

```python
class ViTConsistentDropoutConfig(PolicyArchitecture):
    _dropout_p = 0.2  # Change this value (e.g., 0.1, 0.3, 0.5)
```

### The Data Flow Through the Training Loop

Here's how masks flow through the RL training loop:

```python
# 1. ROLLOUT PHASE (collecting trajectories)
model.train()  # Dropout is active
with torch.no_grad():
    for step in range(num_steps):
        # TensorDict has no masks initially
        td = TensorDict({"env_obs": obs}, batch_size=[num_envs])

        # Forward pass creates NEW masks and stores them in td
        td = model(td)  # td now contains dropout masks

        # Store trajectory in replay buffer (including masks!)
        replay_buffer.add(td)  # Masks are part of the stored data

# 2. TRAINING PHASE (computing gradients)
model.train()  # Still in training mode
for batch in replay_buffer:
    # batch is a TensorDict containing:
    # - obs, actions, rewards, etc.
    # - dropout masks from rollout!

    # Forward pass REUSES the stored masks
    td = model(batch)  # Uses existing masks from batch

    # Compute loss and update
    loss = compute_loss(td)
    loss.backward()
    optimizer.step()
```

**Critical insight**: The replay buffer must store and return the dropout masks as part of the batch. This happens automatically when using TensorDict-based buffers.

### Episode Boundaries

When using consistent dropout, it's important to handle episode boundaries correctly:

- **Reset masks at episode boundaries**: Ensure new masks are sampled when a new episode starts
- **The framework handles this automatically** through the TensorDict reset mechanism
- **Each trajectory gets its own mask**: Masks are sampled per-trajectory, not per-batch

## Training Integration

The consistent dropout implementation integrates seamlessly with the existing training infrastructure through the `get_agent_experience_spec()` mechanism:

### How It Works

1. **Experience Spec Registration**: Both `MLPConsistentDropout` and `ObsPerceiverLatentConsistentDropout` implement `get_agent_experience_spec()` to register their dropout mask requirements.

```python
def get_agent_experience_spec(self) -> Composite:
    spec = Composite()
    for i in range(self._num_layers):
        mask_key = (self.config.name, f"dropout_mask_{i}")
        spec[mask_key] = UnboundedContinuous(
            shape=torch.Size([latent_dim]),
            dtype=torch.float32,
        )
    return spec
```

2. **Automatic Buffer Allocation**: The `Experience` buffer automatically allocates storage for all registered specs, including dropout masks.

3. **TensorDict Flow**: Dropout masks flow naturally through the TensorDict:
   - **Rollout**: Masks are created and stored in TensorDict
   - **Buffer Storage**: TensorDict (with masks) is stored in experience buffer
   - **Training**: Batch from buffer includes masks, which are retrieved and reused

### No Training Code Modifications Needed

**The training code requires NO modifications!** The dropout masks are automatically:
- Registered via `get_agent_experience_spec()`
- Allocated in the experience buffer
- Stored during rollout
- Retrieved during training

The existing training loop in `metta/rl/trainer.py` already handles TensorDict-based data flow, so consistent dropout works out of the box.

## Files Created

1. **Policy Configuration**:
   - `agent/src/metta/agent/policies/vit_consistent_dropout.py` - Main policy architecture

2. **Components with Experience Spec**:
   - `agent/src/metta/agent/components/misc_consistent_dropout.py` - MLP with consistent dropout and `get_agent_experience_spec()`
   - `agent/src/metta/agent/components/obs_enc_consistent_dropout.py` - ObsPerceiverLatent with consistent dropout and `get_agent_experience_spec()`

3. **Recipe**:
   - `experiments/recipes/abes/vit_consistent_dropout.py` - Training recipe for the policy

4. **Documentation**:
   - `docs/vit_consistent_dropout.md` - This comprehensive guide

## Benefits

1. **Stable Training**: Prevents biased gradients that can cause training instability
2. **Better Convergence**: More consistent policy gradient estimates
3. **Regularization**: Still provides dropout's regularization benefits
4. **No Performance Cost**: Same inference speed as standard dropout during rollout

## References

- Hausknecht & Wagener, "On-Policy Deep Reinforcement Learning with Dropout"
- TorchRL ConsistentDropout documentation: https://pytorch.org/rl/
- Original paper: https://ar5iv.org/html/2005.05719

## Testing

To verify the policy works correctly:

```python
from metta.agent.policies.vit_consistent_dropout import ViTConsistentDropoutConfig

# Test instantiation
config = ViTConsistentDropoutConfig()
print(f"Components: {len(config.components)}")
print(f"Dropout probability: {config._dropout_p}")
```

## Future Enhancements

Potential improvements to consider:

1. **Adaptive dropout rates**: Adjust dropout probability during training
2. **Layer-specific dropout**: Different dropout rates for different components
3. **Dropout scheduling**: Gradually reduce dropout over training
4. **Uncertainty estimation**: Leverage consistent dropout for policy uncertainty quantification
