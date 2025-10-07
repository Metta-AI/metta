# Action Compatibility Guide for Metta

This document catalogs all types of changes that can break the action system in Metta, helping developers understand the
impact of their changes on existing trained policies.

## Table of Contents

1. [Action System Overview](#action-system-overview)
2. [Breaking Changes Catalog](#breaking-changes-catalog)
3. [Compatibility Matrix](#compatibility-matrix)
4. [Migration Strategies](#migration-strategies)
5. [Testing for Compatibility](#testing-for-compatibility)

## Action System Overview

### Action Structure

Metta actions are represented by a **single discrete index**. Each index corresponds to a fully qualified action variant
such as `move_north`, `rotate_west`, or `attack_3`. Verb/argument combinations are flattened during environment
construction, so policies only emit a scalar `action_id` per agent.

> **Note:** Older builds exposed actions as `(type, argument)` pairs. Those multi-field tensors are now deprecated. When
> replaying historical data, use the migration helpers in `mettagrid.test_support.actions` to map legacy tuples onto the
> new enumerated action ids.

### Action Validation Flow

```
1. Action Index Validation: 0 <= action_id < num_action_handlers
2. Action Space Validation: ensure sampled indices are within `env.action_space.n`
3. Agent State Validation: Check if agent is frozen
4. Resource Validation: Check required/consumed resources
5. Action Execution: Attempt the action
```

### Key Components

- **MettaGrid::\_step()**: Main action processing loop
- **ActionHandler**: Base class for all actions
- **\_handle_invalid_action()**: Error handling for invalid actions
- **action_names()**: Maps action names to indices

## Breaking Changes Catalog

### 1. Action Index Changes

**Description**: Modifying the order in which actions are registered changes their numeric indices.

**Example**:

```yaml
# Before
actions:
  noop: {...}      # index 0
  move: {...}      # index 1
  rotate: {...}    # index 2

# After (BREAKING)
actions:
  move: {...}      # index 0 (was 1)
  noop: {...}      # index 1 (was 0)
  rotate: {...}    # index 2
```

**Impact**: Trained policies will execute wrong actions **Detection**: Action success rates drop dramatically **Stats
Tracked**: `action.invalid_type`

### 2. Action Variant Count Changes

**Description**: Changing how many concrete variants are generated for a logical verb.

**Example**:

```yaml
# Before
actions:
  move:
    allow_diagonals: false  # Only N, S, E, W -> 4 move_* entries

# After (BREAKING)
actions:
  move:
    allow_diagonals: true   # Adds NE/NW/SE/SW -> 8 move_* entries
```

**Impact**:

- Removing variants: Formerly valid action ids now fall outside the handler list
- Adding variants: Action ids shift after the insertion point; trained policies must be remapped

**Detection**: `action.invalid_type` errors increase **Stats Tracked**: `action.invalid_type`, `action.<name>.failed`

### 3. Action Removal or Renaming

**Description**: Removing an action or changing its name in configuration.

**Example**:

```yaml
# Before
actions:
  attack: {...}

# After (BREAKING)
actions:
  # attack removed or renamed to combat
```

**Impact**:

- Removal: Action indices shift, causing wrong actions
- Renaming: Action becomes unregistered

**Detection**: `Unknown action` errors during initialization **Stats Tracked**: `action.invalid_type`

### 4. Resource Requirement Changes

**Description**: Modifying required or consumed resources for actions.

**Example**:

```cpp
// Before
required_resources: {ore: 1}
consumed_resources: {ore: 1}

// After (BREAKING)
required_resources: {ore: 2, energy: 1}
consumed_resources: {ore: 2}
```

**Impact**: Actions fail due to insufficient resources **Detection**: Action success rate drops **Stats Tracked**:
`action.<name>.failed`

### 5. Agent State Conflicts

**Description**: Changes to agent state that prevent action execution.

**Example**: Frozen duration changes

```cpp
// Frozen agents cannot perform actions
if (actor->frozen != 0) {
    return false;
}
```

**Impact**: Actions fail silently **Stats Tracked**: `status.frozen.ticks`, `status.frozen.ticks.<group>`

### 6. Action Priority Changes

**Description**: Modifying action execution priority affects order of processing.

**Example**:

```cpp
// Attack has priority 1, others have priority 0
// Actions execute from highest to lowest priority
```

**Impact**: Action conflicts resolved differently **Detection**: Subtle behavior changes in multi-agent scenarios

### 7. Inventory Item Index Changes

**Description**: Changing the order or IDs of inventory items that actions depend on.

**Example**:

```yaml
# Before
resource_names: [ore, wood, gold]  # ore=0, wood=1, gold=2

# After (BREAKING)
resource_names: [wood, ore, gold]  # wood=0, ore=1, gold=2
```

**Impact**: Resource checks use wrong items **Detection**: Resource-based actions fail unexpectedly

### 8. Action Handler Implementation Changes

**Description**: Modifying the internal logic of action handlers.

**Example**:

```cpp
// Move action now checks for walls differently
bool Move::_handle_action(Agent* actor, ActionArg arg) {
    // New collision detection logic
}
```

**Impact**: Actions succeed/fail in different situations **Detection**: Behavior changes without explicit errors

### 9. Special Action Cases

**Attack Action Duplication**:

```cpp
// Attack action creates TWO handlers
_action_handlers.push_back(std::make_unique<Attack>(...));
_action_handlers.push_back(std::make_unique<AttackNearest>(...));
```

**Impact**: Attack takes up two action indices

### 10. Action Space Flattening

**Description**: Legacy builds exposed a `MultiDiscrete([num_verbs, max_arg + 1])` action space. Current builds flatten every verb/argument combination into a `gymnasium.spaces.Discrete(num_variants)` space.

**Example**:

```python
# Before (legacy two-field actions)
action_space = gymnasium.spaces.MultiDiscrete([len(verbs), max_arg + 1])

# After (current single-index actions)
action_space = gymnasium.spaces.Discrete(len(env.action_names()))
# env.action_names() -> ["noop", "move_north", "move_south", "attack_0", ...]
```

**Impact**: Actor heads emit a single logit vector. Use `env.action_names()` to interpret sampled indices. **Detection**: Runtime shape errors in policy

## Compatibility Matrix

| Change Type              | Requires Retraining | Backward Compatible | Forward Compatible | Migration Available |
| ------------------------ | ------------------- | ------------------- | ------------------ | ------------------- |
| Action reordering        | ✅ Yes              | ❌ No               | ❌ No              | ⚠️ Possible         |
| Variant count increase   | ⚠️ Partial          | ✅ Yes              | ❌ No              | ✅ Yes              |
| Variant count decrease   | ✅ Yes              | ❌ No               | ❌ No              | ❌ No               |
| Action removal           | ✅ Yes              | ❌ No               | ❌ No              | ⚠️ Possible         |
| Action addition          | ❌ No               | ✅ Yes              | ✅ Yes             | ✅ Yes              |
| Resource changes         | ⚠️ Depends          | ⚠️ Partial          | ⚠️ Partial         | ✅ Yes              |
| Priority changes         | ⚠️ Depends          | ✅ Yes              | ✅ Yes             | ✅ Yes              |
| Item reordering          | ✅ Yes              | ❌ No               | ❌ No              | ⚠️ Possible         |

### Legend

- ✅ Yes: Fully supported
- ❌ No: Not supported
- ⚠️ Partial/Depends: Case-by-case basis

## Migration Strategies

### 1. Action Index Remapping

**Strategy**: Maintain a mapping between old and new action indices.

```python
# Flatten legacy (type, arg) pairs into a single id
def legacy_flatten(action_type: int, action_arg: int, max_args: list[int]) -> int:
    offset = sum(max_args[i] + 1 for i in range(action_type))
    return offset + action_arg

# Remap flattened ids onto the new enumerated action list
ACTION_REMAP = {
    0: 0,   # noop -> noop
    1: 4,   # move,dir=0 -> move_north
    2: 5,   # move,dir=1 -> move_south
}

def migrate_action(old_action, max_args):
    action_type, action_arg = old_action
    legacy_id = legacy_flatten(action_type, action_arg, max_args)
    return ACTION_REMAP.get(legacy_id, legacy_id)
```

### 2. Gradual Resource Requirement Changes

**Strategy**: Implement grace periods or fallbacks.

```cpp
// Check both old and new requirements during transition
bool has_resources = check_new_requirements(actor) ||
                    check_legacy_requirements(actor);
```

### 3. Action Space Padding

**Strategy**: Pad action space to maintain dimensions.

```python
# Add dummy actions to maintain space
if len(action_names) < expected_actions:
    for i in range(expected_actions - len(action_names)):
        action_names.append(f"dummy_{i}")
```

### 4. Version-Aware Policies

**Strategy**: Store environment version with policies.

```python
metadata = {
    "env_version": "1.2.0",
    "action_mapping": {...},
    "resource_mapping": {...}
}
```

## Testing for Compatibility

### 1. Action Validation Tests

```python
def test_action_compatibility():
    # Test all valid action combinations
    for action_type in range(num_actions):
        for arg in range(max_args[action_type] + 1):
            obs, reward, done, info = env.step([action_type, arg])
            assert env.action_success()[0], f"Valid action {action_type},{arg} failed"
```

### 2. Policy Regression Tests

```python
def test_policy_behavior():
    # Load saved policy
    policy = load_policy("trained_model.pt")

    # Test on reference scenarios
    for scenario in test_scenarios:
        env.reset(scenario)
        total_reward = run_episode(env, policy)
        assert total_reward > threshold, f"Policy performance degraded"
```

### 3. Breaking Change Detection

```python
def detect_breaking_changes(old_config, new_config):
    issues = []

    # Check action order
    old_actions = list(old_config['actions'].keys())
    new_actions = list(new_config['actions'].keys())
    if old_actions != new_actions[:len(old_actions)]:
        issues.append("Action order changed")

    # Check max args
    for action in old_actions:
        if action in new_actions:
            old_max = get_max_arg(old_config, action)
            new_max = get_max_arg(new_config, action)
            if new_max < old_max:
                issues.append(f"{action} max_arg reduced")

    return issues
```

### 4. Monitoring Metrics

Key metrics to track for compatibility issues:

- `action.invalid_type` - Spike indicates index mismatch
- `action.invalid_arg` - Spike indicates max_arg issues
- `action.<name>.failed` - Drop in success rate
- `action.failure_penalty` - Increased penalties
- Episode rewards - Sudden drops indicate breaking changes

## Best Practices

1. **Version Control**: Always version your action configurations
2. **Compatibility Tests**: Run regression tests before deploying changes
3. **Gradual Rollout**: Test changes with a subset of agents first
4. **Documentation**: Document all action space changes in release notes
5. **Metadata Storage**: Save action mappings with trained policies
6. **Monitoring**: Set up alerts for action failure rate changes

## Future Compatibility Work

1. **Action Aliasing**: Support multiple names for the same action
2. **Dynamic Action Spaces**: Allow runtime action space modifications
3. **Compatibility Layers**: Automatic translation between versions
4. **Policy Adapters**: Wrapper classes for legacy policies
