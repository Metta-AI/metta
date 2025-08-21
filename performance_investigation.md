# Performance Investigation - Arena Training Regression After Dehydration

## 🔴 CRITICAL Problem Statement
- **PRIMARY ISSUE**: Agents stuck at ~0.5 heart.get vs 15 heart.get before dehydration
- **SECONDARY ISSUE**: SPS dropped from ~450k to ~350k after dehydration
- The heart.get values flatline and don't improve over training!
- **Current command**: `./tools/run.py experiments.recipes.arena_easy_shaped.train --args run=relh.easy_shaped.820.5`

## All Fixes Applied to Date

### ✅ Fixed Critical Bugs

#### 1. Environment Recreation on Every Reset (CRITICAL FIX)
**File**: `mettagrid/src/metta/mettagrid/core.py`
**Issue**: Environment was completely recreated on every reset, losing episode rewards
**Fix**: Only recreate when necessary (if None or config changed)
```python
def reset(self, seed: Optional[int] = None):
    if self.__c_env_instance is None or self._config_changed:
        self._create_c_env()
        self._update_core_buffers()
        self._config_changed = False
```

#### 2. desync_episodes Implementation Changed
**File**: `mettagrid/src/metta/mettagrid/mettagrid_env.py`
**Issue**: New implementation truncated episodes early instead of changing max_steps
**Fix**: Restored old behavior - modify max_steps for first episode only
```python
if self._is_training and env_cfg.desync_episodes and self._first_episode:
    env_cfg.game.max_steps = int(np.random.randint(1, self._original_max_steps + 1))
```

#### 3. CurriculumEnv Performance Fix
**File**: `metta/cogworks/curriculum/curriculum_env.py`
**Change**: Replaced `__getattribute__` with `__getattr__` to reduce overhead

#### 4. Arena Easy Shaped Recipe Configuration
**File**: `experiments/recipes/arena_easy_shaped.py`
**Fixed**:
- Added 20 blocks back to the map (were missing)
- Set walls to 20 instead of 10
- Removed lasery and armory buildings
- Set altar to require only 1 battery_red (was 3)
- Added all shaped rewards (ore: 0.1, battery: 0.8, heart: 1.0)

### ✅ Restored Configuration Defaults

All pre-dehydration defaults restored in `metta/rl/trainer_config.py`:

| Parameter | Old (Pre-Dehydration) | Was Broken | Now Fixed |
|-----------|----------------------|------------|-----------|
| Checkpoint interval | 50 epochs | 5 epochs | ✅ 50 |
| WandB checkpoint interval | 50 epochs | 5 epochs | ✅ 50 |
| Evaluate remote | True | False | ✅ True |
| Evaluate local | False | True | ✅ False |
| V-trace rho clip | 1.0 | 1.0 | ✅ Already correct |
| V-trace c clip | 1.0 | 1.0 | ✅ Already correct |
| Prioritized replay alpha | 0.0 | 0.0 | ✅ Already correct |
| Grad mean variance interval | 0 | 0 | ✅ Already correct |
| Profiler interval | 10000 | 0 | ✅ 10000 |

### ✅ Verified Non-Issues

1. **Hyperparameter Scheduler**: Was already disabled pre-dehydration (code commented out)
2. **PPO hyperparameters**: All correct (clip=0.1, ent=0.0021, gae=0.916, gamma=0.977)
3. **Optimizer settings**: Correct (lr=0.000457, beta1=0.9, beta2=0.999)
4. **Batch sizes**: Correct (batch=524288, minibatch=16384)

## Major Architectural Differences Found

### 1. Training Invocation Changed
**Old**: `./tools/train.py` with Hydra/YAML configs
**New**: `./tools/run.py experiments.recipes.arena_easy_shaped.train` with Python configs

### 2. CurriculumEnv Wrapper Added Everywhere
**Old**: Environment created directly
**New**: ALL environments wrapped with `CurriculumEnv`
- Calls `set_env_config()` after every episode
- Sets `_config_changed = True`, potentially triggering recreation
- Adds overhead to every operation

### 3. Worker Count Calculation Changed
**Old**: Rounded down to nearest power of 2
```python
while num_workers * 2 <= ideal_workers:
    num_workers *= 2
```
**New**: Direct calculation without rounding
```python
cfg.trainer.rollout_workers = max(1, ideal_workers)
```

### 4. Configuration System
**Old**: YAML-based with Hydra
**New**: Python-based with Tool abstraction layer

## Deep Investigation Findings

### Agent Creation and Initialization
- Agent is created as `MettaAgent(metta_grid_env, system_cfg, agent_cfg)`
- Uses ComponentFast architecture (same as before)
- Feature normalizations passed correctly from environment
- Policy initialized with correct observation/action spaces

### Reward Flow Analysis
1. Rewards configured in Python: `env_cfg.game.agent.rewards.inventory.ore_red = 0.1`
2. Passed via `model_dump()` to C++ config converter
3. Extracted correctly in `mettagrid_c_config.py` lines 50-57
4. Set as `resource_rewards` in agent config

### Experience Buffer and PPO
- Experience buffer implementation unchanged
- PPO loss computation standard and correct
- Advantage computation using same CUDA kernels
- Batch accumulation logic unchanged

### Environment Management
- `training_env_id` handling unchanged
- Episode stats collection at episode end
- Buffer allocation with zero_copy unchanged
- Async factor still set to 2

## Remaining Suspects

### 1. CurriculumEnv Wrapper Impact (HIGH PRIORITY)
Even though it "shouldn't be a big thing", it's the most significant architectural change:
- Not present in old system at all
- Wraps EVERY environment
- Calls `set_env_config()` after every episode
- Could interfere with reward accumulation or state management

### 2. Worker Count Power-of-2 Rounding
Old system had specific reason for power-of-2 rounding - might affect:
- Memory alignment
- CPU cache efficiency
- Parallel processing patterns

### 3. Unknown Differences in C++ Layer
Haven't fully verified:
- Episode reward accumulation in C++
- Resource generation timing
- Action processing

## What We Haven't Tried Yet

1. **Remove CurriculumEnv wrapper entirely** - Test if this is the root cause
2. **Restore power-of-2 worker rounding** - Could affect performance
3. **Add comprehensive C++ logging** - Verify rewards are being generated
4. **Test with navigation environment** - Verify if simpler env works

## Diagnostic Tools Created

### diagnose_training.py
Created diagnostic script to:
- Verify environment configuration
- Run steps and check reward generation
- Test environment recreation
- Monitor shaped rewards

## Next Steps

1. **Most Important**: Test without CurriculumEnv wrapper to isolate its impact
2. Add logging to verify rewards are actually being generated in C++
3. Monitor first 1000 steps to see if agents ever get rewards
4. Check if navigation environment works (simpler reward structure)

## Summary

Despite fixing multiple critical bugs and restoring all configuration defaults, arena training still fails. The most significant remaining difference is the CurriculumEnv wrapper that wasn't present in the old system. This wrapper:
- Is applied to every environment
- Calls `set_env_config()` after every episode  
- May interfere with the complex resource chain in arena

The fact that navigation works but arena doesn't suggests the issue is specific to complex reward chains, not general training.