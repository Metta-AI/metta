# PyTorchAgentMixin Usage Guide

## Overview

The `PyTorchAgentMixin` class provides shared functionality for all PyTorch-based agents in the Metta project. It ensures consistency across implementations and simplifies the creation of new PyTorch agents by handling common requirements that MettaAgent expects.

## What the Mixin Provides

### 1. Configuration Management
- Handles `clip_range` and `analyze_weights_interval` parameters
- Manages additional kwargs for future extensions
- Provides consistent logging of configuration

### 2. Weight Clipping
- Implements `clip_weights()` method matching ComponentPolicy
- Called by trainer after optimizer steps
- Prevents gradient explosion during training

### 3. TensorDict Field Management
- Sets critical `td["bptt"]` and `td["batch"]` fields
- Required for LSTM state management
- Ensures experience buffer compatibility

### 4. Action Conversion
- Overrides `_convert_action_to_logit_index()` with compensating formula
- Overrides `_convert_logit_index_to_action()` using policy's tensors
- Matches ComponentPolicy's behavior exactly

### 5. Training/Inference Mode Handling
- `handle_training_mode()`: Processes training with proper TD reshaping
- `handle_inference_mode()`: Handles action sampling during inference
- Manages flattening and reshaping of TensorDicts

## How to Use the Mixin

### Step 1: Import the Mixin

```python
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin
from metta.agent.pytorch.base import LSTMWrapper
```

### Step 2: Inherit from Both Mixin and Base Class

```python
class MyAgent(PyTorchAgentMixin, LSTMWrapper):
    # Mixin must come before LSTMWrapper for proper method resolution
```

### Step 3: Initialize the Mixin in __init__

```python
def __init__(self, env, policy=None, input_size=128, hidden_size=128, 
             num_layers=2, **kwargs):
    # Extract mixin parameters
    mixin_params = self.extract_mixin_params(kwargs)
    
    # Initialize parent classes
    super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)
    
    # Initialize mixin
    self.init_mixin(**mixin_params)
```

### Step 4: Use Mixin Methods in forward()

```python
@torch._dynamo.disable
def forward(self, td: TensorDict, state=None, action=None):
    observations = td["env_obs"]
    
    # Use mixin to set TensorDict fields
    B, TT = self.set_tensordict_fields(td, observations)
    
    # ... agent-specific encoding and LSTM forward ...
    
    logits_list, value = self.policy.decode_actions(flat_hidden, B * TT)
    
    # Use mixin for mode-specific processing
    if action is None:
        td = self.handle_inference_mode(td, logits_list, value)
    else:
        td = self.handle_training_mode(td, action, logits_list, value)
    
    return td
```

## Example: Before and After

### Before (Fast.py) - ~170 lines in forward() and related methods

```python
class Fast(LSTMWrapper):
    def __init__(self, env, ..., clip_range=0, analyze_weights_interval=300, **kwargs):
        # Manual parameter handling
        super().__init__(...)
        self.clip_range = clip_range
        self.analyze_weights_interval = analyze_weights_interval
        if kwargs:
            logger.info(f"[DEBUG] Additional config parameters: {kwargs}")
    
    def clip_weights(self):
        # 10 lines of weight clipping code
        
    def forward(self, td, state=None, action=None):
        # 15 lines of TensorDict field setup
        # 30 lines of training mode handling
        # 20 lines of inference mode handling
        
    def _convert_action_to_logit_index(self, ...):
        # 7 lines of conversion logic
        
    def _convert_logit_index_to_action(self, ...):
        # 3 lines of conversion logic
```

### After (FastRefactored.py) - ~40 lines in forward()

```python
class FastRefactored(PyTorchAgentMixin, LSTMWrapper):
    def __init__(self, env, ..., **kwargs):
        mixin_params = self.extract_mixin_params(kwargs)
        super().__init__(...)
        self.init_mixin(**mixin_params)
    
    # clip_weights() provided by mixin
    
    def forward(self, td, state=None, action=None):
        observations = td["env_obs"]
        B, TT = self.set_tensordict_fields(td, observations)
        
        # Agent-specific logic only
        hidden = self.policy.encode_observations(observations, state)
        # ... LSTM forward ...
        logits_list, value = self.policy.decode_actions(flat_hidden, B * TT)
        
        # Use mixin for mode handling
        if action is None:
            td = self.handle_inference_mode(td, logits_list, value)
        else:
            td = self.handle_training_mode(td, action, logits_list, value)
        return td
    
    # Action conversion methods provided by mixin
```

## Benefits of Using the Mixin

### 1. **Consistency**
- All PyTorch agents behave identically for common operations
- Reduces bugs from inconsistent implementations
- Ensures compatibility with MettaAgent and trainer

### 2. **Maintainability**
- Bug fixes in common logic only need to be made once
- New features can be added to all agents simultaneously
- Clearer separation between agent-specific and shared logic

### 3. **Simplicity**
- New agents are easier to create
- Less boilerplate code
- Focus on agent-specific innovations

### 4. **Documentation**
- Single source of truth for common functionality
- Better understanding of what's shared vs unique
- Easier onboarding for new developers

## Critical Implementation Notes

### Action Conversion Formula
The mixin uses the compensating formula that matches ComponentPolicy:
```python
return action_type_numbers + cumulative_sum + action_params
```

This differs from MettaAgent's base implementation which uses:
```python
return cumulative_sum + action_params  # Missing action_type_numbers
```

The cumsum calculation in MettaAgent is technically wrong, but ComponentPolicy compensates with its formula. Both must be kept in sync!

### TensorDict Fields
The `td["bptt"]` and `td["batch"]` fields are **critical** for:
- LSTM components in ComponentPolicy
- Experience buffer integration
- Proper tensor reshaping during training

Without these fields, training will fail or produce incorrect results.

### Method Resolution Order
When using the mixin, it must come before the base class in inheritance:
```python
class MyAgent(PyTorchAgentMixin, LSTMWrapper):  # Correct
class MyAgent(LSTMWrapper, PyTorchAgentMixin):  # Wrong - mixin methods won't override
```

## Migration Guide for Existing Agents

To migrate an existing PyTorch agent to use the mixin:

1. **Add mixin to inheritance**: `class MyAgent(PyTorchAgentMixin, LSTMWrapper)`
2. **Update __init__**: Use `extract_mixin_params()` and `init_mixin()`
3. **Remove duplicate methods**: Delete `clip_weights()`, action conversion methods
4. **Update forward()**: Use mixin's `set_tensordict_fields()` and mode handlers
5. **Test thoroughly**: Ensure behavior matches original implementation

## Creating New Agents

When creating a new PyTorch agent:

1. **Start with the mixin**: Inherit from `PyTorchAgentMixin` and appropriate base
2. **Focus on unique logic**: Implement only agent-specific encoding/decoding
3. **Use mixin utilities**: Leverage all provided methods
4. **Document differences**: Clearly note what makes your agent unique

## Testing

Agents using the mixin should be tested for:
- Configuration parameter passing
- Weight clipping activation
- TensorDict field presence
- Action conversion correctness
- Training stability

## Future Enhancements

The mixin can be extended to handle:
- Additional configuration parameters
- More sophisticated state management
- Performance metrics collection
- Debugging utilities
- Automatic validation

By using `PyTorchAgentMixin`, we ensure all PyTorch agents in the Metta project maintain consistency, compatibility, and correctness while reducing code duplication and maintenance burden.