# SafeModule Usage Guide

## Overview
This guide explains when and where to use `SafeModule` wrappers in the MettaAgent architecture, based on TorchRL patterns and SafeRL research.

## Key Principle
**Apply safety wrappers at critical decision points** - especially where the agent interacts with the environment or where numerical instability can cause system failure.

## Primary Use Cases

### 1. Actor/Policy Networks (CRITICAL)
The most important use case is wrapping **policy networks** that output actions:

```python
# For continuous action spaces with bounded actions
safe_policy = SafeModule(
    policy_module,
    action_bounds=(-1.0, 1.0),  # Clip actions to valid range
    nan_check=True
)

# In ComponentContainer
components.register_component("policy", safe_policy, deps=["obs_processor"])
```

**Why**: Policy outputs directly control environment actions. Invalid actions can:
- Violate environment constraints
- Cause simulation crashes  
- Lead to unsafe real-world behavior

### 2. Value Networks (IMPORTANT)
Value networks can explode or output NaN, especially during early training:

```python
safe_value = SafeModule(
    value_module,
    nan_check=True  # Catch value function explosions
)
```

**Why**: Value function instability can destabilize learning and cause training divergence.

### 3. Environment Interface Components
Wrap any component that directly interfaces with the environment:

```python
# Final action processing before environment
action_processor = SafeModule(
    action_head,
    action_bounds=env.action_space.bounds,
    nan_check=True
)
```

### 4. Multi-Environment Scenarios
When recycling models across different environments:

```python
# Universal policy that adapts to different environments
universal_policy = SafeModule(
    policy_net,
    action_bounds=None,  # Will be set per environment
    nan_check=True
)

# Later, update bounds for specific environment
universal_policy.action_bounds = current_env.action_spec.bounds
```

## What NOT to Wrap

### Low-Risk Components
- **Observation processing**: Usually stable and bounded by environment
- **Internal feature extraction**: Less critical for system safety
- **Auxiliary networks**: Non-critical components like attention mechanisms

```python
# These typically don't need safety wrappers
obs_processor = LinearModule(64, 32, "observation", "features")
attention_layer = AttentionModule("features", "attended_features")
```

## MettaAgent Integration Pattern

### Recommended Architecture
```python
class MettaAgent:
    def __init__(self, env, ...):
        self.components = ComponentContainer()
        
        # 1. Wrap CRITICAL policy component
        safe_policy = SafeModule(
            policy_network, 
            action_bounds=env.action_space.bounds,
            nan_check=True
        )
        
        # 2. Wrap value networks for stability
        safe_value = SafeModule(
            value_network,
            nan_check=True
        )
        
        # 3. Register with dependencies
        self.components.register_component("obs_processor", obs_module)
        self.components.register_component("policy", safe_policy, deps=["obs_processor"])
        self.components.register_component("value", safe_value, deps=["obs_processor"])
```

### Safety Priority Levels
1. **CRITICAL**: Policy/Actor networks → Always wrap
2. **HIGH**: Value/Critic networks → Usually wrap  
3. **MEDIUM**: Environment interfaces → Wrap if directly connected
4. **LOW**: Internal processing → Usually no wrapping needed

## TorchRL Integration
Following TorchRL's SafeModule pattern:

```python
# TorchRL's recommended approach
policy = Actor(
    base_module,
    in_keys=["observation"], 
    out_keys=["action"],
    spec=action_spec,
    safe=True  # Uses SafeModule internally
)

# Our equivalent MettaModule approach  
safe_policy = SafeModule(
    MettaModule(base_module, ["observation"], ["action"]),
    action_bounds=action_spec.bounds,
    nan_check=True
)
```

## Safety Configuration Examples

### Conservative (Development/Testing)
```python
# Wrap everything critical
safe_policy = SafeModule(policy, action_bounds=(-1, 1), nan_check=True)
safe_value = SafeModule(value, nan_check=True) 
safe_critic = SafeModule(critic, nan_check=True)
```

### Production (Performance Optimized)
```python
# Wrap only the most critical components
safe_policy = SafeModule(policy, action_bounds=env.bounds, nan_check=True)
# Leave stable components unwrapped for performance
value_network = ValueModule(...)  # No wrapper
```

### Debugging Mode
```python
# Enable all safety checks for debugging
safe_component = SafeModule(
    component,
    action_bounds=bounds,
    nan_check=True  # Strict NaN/Inf checking
)
```

## Best Practices

1. **Start Conservative**: Wrap all critical components during development
2. **Profile Performance**: Remove unnecessary wrappers for production
3. **Environment-Specific**: Adjust action bounds per environment
4. **Monitor Safety**: Log safety violations for analysis
5. **Gradual Relaxation**: Remove wrappers only after thorough testing

## Summary

**Rule of Thumb**: If a component failure could cause:
- Environment crashes
- Training instability  
- Unsafe actions
- System-wide failure

→ **Wrap it with SafeModule**

The SafeModule provides a crucial safety net that prevents numerical instabilities and constraint violations from propagating through the system. 