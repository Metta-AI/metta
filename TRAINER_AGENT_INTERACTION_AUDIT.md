# Trainer-Agent Interaction Audit

## Critical Finding: PyTorch Policies Missing Configuration

### The Problem

In `MettaAgent._create_policy()` (lines 105-123), there's a critical difference in how policies are created:

**ComponentPolicy (agent=fast):**
```python
policy = ComponentPolicy(
    obs_space=self.obs_space,
    obs_width=self.obs_width,
    obs_height=self.obs_height,
    action_space=self.action_space,
    feature_normalizations=self.feature_normalizations,
    device=system_cfg.device,
    cfg=agent_cfg,  # <-- Gets full configuration!
)
```

**PyTorch Policy (py_agent=fast):**
```python
AgentClass = agent_classes[agent_cfg.agent_type]
policy = AgentClass(env=env)  # <-- ONLY gets env!
```

### Missing Configuration Parameters

PyTorch policies are NOT receiving:
1. `clip_range` from agent_cfg
2. `analyze_weights_interval` from agent_cfg
3. Any other custom configuration parameters
4. System configuration (device, etc.)

### Impact on Training

1. **Weight Clipping Broken**: 
   - Fast.__init__ defaults `clip_range=0` (no clipping)
   - ComponentPolicy gets `clip_range` from YAML
   - trainer.py calls `policy.clip_weights()` at line 420
   - PyTorch policies never clip weights!

2. **Configuration Mismatch**:
   - ComponentPolicy reads all YAML settings
   - PyTorch policies use hardcoded defaults

## Training Loop Analysis

### Key Interaction Points

1. **Policy Creation** (checkpoint_manager.py:227):
   ```python
   new_policy_record.policy = MettaAgent(metta_grid_env, system_cfg, agent_cfg)
   ```

2. **Policy Initialization** (trainer.py:188-193):
   ```python
   initialize_policy_for_environment(
       policy_record=latest_saved_policy_record,
       metta_grid_env=metta_grid_env,
       device=device,
       restore_feature_mapping=True,
   )
   ```

3. **Training Step** (trainer.py:410-420):
   ```python
   optimizer.zero_grad()
   loss.backward()
   torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer_cfg.ppo.max_grad_norm)
   optimizer.step()
   policy.clip_weights()  # <-- Calls MettaAgent.clip_weights()
   ```

### MettaAgent.clip_weights() Behavior

From metta_agent.py:
```python
def clip_weights(self):
    """Clip weights to prevent large updates."""
    if self.policy is not None and hasattr(self.policy, "clip_weights"):
        # Use policy's custom implementation if available
        self.policy.clip_weights()
    elif self.policy is not None and self.clip_range > 0:
        # Default implementation for any PyTorch policy
        for module in self.policy.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    module.weight.data.clamp_(-self.clip_range, self.clip_range)
```

**Problem**: 
- MettaAgent sets `self.clip_range = agent_cfg.get("clip_range", 0)`
- But PyTorch policies also have their own `clip_range` attribute
- Confusion about which `clip_range` is used where

## Required Fixes

### Fix 1: Pass Configuration to PyTorch Policies

```python
# In MettaAgent._create_policy()
if agent_cfg.get("agent_type") in agent_classes:
    AgentClass = agent_classes[agent_cfg.agent_type]
    # Pass configuration parameters
    policy = AgentClass(
        env=env,
        clip_range=agent_cfg.get("clip_range", 0),
        analyze_weights_interval=agent_cfg.get("analyze_weights_interval", 300),
        # Any other config params...
    )
```

### Fix 2: Standardize Constructor Signatures

All PyTorch policies should accept:
```python
def __init__(self, env, clip_range=0, analyze_weights_interval=300, **kwargs):
```

### Fix 3: Clarify Weight Clipping Ownership

Either:
- A) MettaAgent handles all weight clipping (current attempt)
- B) Policies handle their own weight clipping (cleaner)

## Other Observations

### 1. Memory Reset Timing
- `policy.reset_memory()` called at line 303 before rollout
- `policy.reset_memory()` called at line 384 before each minibatch
- PyTorch policies now properly handle this via base class

### 2. Distributed Wrapping
- Line 183: Policy wrapped in DistributedDataParallel
- DistributedMettaAgent forwards calls via `__getattr__`
- Should work for both ComponentPolicy and PyTorch policies

### 3. Compilation
- Line 177: `torch.compile()` applied if configured
- Should work for both policy types

### 4. Optimizer Creation
- Lines 223-241: Creates Adam or Muon optimizer
- Uses `policy.parameters()` - works for both types

## Summary

The main issue is that **PyTorch policies are not receiving their configuration parameters** from agent_cfg. This causes:

1. Weight clipping to be disabled (clip_range=0)
2. Other configuration mismatches
3. Potential training instability

The fix is straightforward: pass agent_cfg parameters to PyTorch policy constructors.