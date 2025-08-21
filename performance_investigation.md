# Performance Investigation - SPS and Training Performance Issues

## 🔴 CRITICAL Problem Statement
- **PRIMARY ISSUE**: Agents stuck at ~0.5 heart.get vs 15 heart.get before dehydration
- **SECONDARY ISSUE**: SPS dropped from ~450k to ~350k after dehydration commit
- The heart.get values flatline and don't improve over training!

## Proposed Fixes

### ✅ Completed Fixes

#### 1. CurriculumEnv Performance Fix
**File**: `metta/cogworks/curriculum/curriculum_env.py`
**Status**: ✅ Implemented
**Change**: Replaced `__getattribute__` with `__getattr__` and explicit method overrides
**Impact**: Should improve SPS by reducing Python overhead on every attribute access

#### 2. Arena Easy Shaped Recipe Environment Fix  
**File**: `experiments/recipes/arena_easy_shaped.py`
**Status**: ✅ Implemented
**Changes**:
- Added 20 blocks back to the map (were missing)
- Set walls to 20 instead of 10
- Removed lasery and armory buildings
- Set altar to require only 1 battery_red (was 3)
- Added all shaped rewards (ore: 0.1, battery: 0.8, etc.)

### ❓ Proposed But Not Yet Verified

#### 3. Worker Count Power-of-2 Rounding
**File**: `metta/tools/train.py` line ~162
**Status**: ❌ Not implemented (you said it's unreasonable)
**Original Proposal**: Round worker count down to nearest power of 2
```python
# Current code:
ideal_workers = (os.cpu_count() // 2) // torch.cuda.device_count()
cfg.trainer.rollout_workers = max(1, ideal_workers)

# Proposed (but disputed):
num_workers = 1
while num_workers * 2 <= ideal_workers:
    num_workers *= 2
cfg.trainer.rollout_workers = max(1, num_workers)
```

#### 4. Checkpoint Interval Adjustment
**File**: `experiments/recipes/arena_easy_shaped.py`
**Status**: ❌ Not implemented (you said it probably doesn't matter)
**Note**: Changed from time-based (seconds) to epoch-based intervals
- Old: 60 seconds checkpoint, 300 seconds evaluate
- New: 5 epochs checkpoint, 50 epochs evaluate
- Recipe currently uses: 50 epochs for both

#### 5. CurriculumEnv Config Timing
**File**: `metta/cogworks/curriculum/curriculum_env.py`
**Status**: ❌ Not implemented
**Issue**: `set_env_config()` called immediately on episode end, before reset
**Proposed Fix**: Delay config change until reset
```python
# Store config for next reset instead of applying immediately
self._pending_config = self._current_task.get_env_cfg()
```

## Key Environment Differences Found

### Old Default (`/env/mettagrid/arena/basic_easy_shaped`)
- **Objects**: 10 mines, 5 generators, 5 altars, 20 blocks, 20 walls
- **Rewards**: Shaped (ore: 0.1, battery: 0.8, laser: 0.5, armor: 0.5, blueprint: 0.5, heart: 1)
- **Altar**: 1 battery → 1 heart (easy)
- **Combat**: Disabled (laser cost: 100)

### New Default (`make_arena()`)
- **Objects**: 10 mines, 5 generators, 5 altars, 0 blocks, 10 walls, 1 lasery, 1 armory
- **Rewards**: Only heart: 1 (no shaped rewards)
- **Altar**: 3 batteries → 1 heart (hard)
- **Combat**: Enabled by default (laser cost: 1)

## Questions to Verify

1. **Is the recipe actually being used?**
   - Need to verify curriculum is created from arena_easy_shaped
   - Check if shaped rewards are active during training

2. **Is set_env_config() causing issues?**
   - Check if environment is being rebuilt during episodes
   - Verify rewards are collected properly

3. **Are checkpoints too frequent?**
   - Monitor actual checkpoint frequency in epochs vs time
   - Check I/O overhead

## All Investigated Issues & Concerns

### Environment & Curriculum Issues
1. **Default environment completely wrong** (VERIFIED ISSUE)
   - Missing 20 blocks (navigation obstacles)
   - Only 10 walls instead of 20
   - Has combat buildings (lasery, armory) that shouldn't be there
   - Altar needs 3 batteries instead of 1 (3x harder!)
   - NO shaped rewards (only heart=1)

2. **CurriculumEnv wrapper overhead**
   - Was using `__getattribute__` intercepting every call (FIXED)
   - Still wraps environment adding indirection
   - Calls `set_env_config()` mid-episode potentially corrupting state

3. **Environment reset/config timing**
   - `set_env_config()` rebuilds map_builder during episode
   - Could be losing reward information
   - DesyncEpisodes feature may behave differently

### Training Loop & PPO Issues
4. **Checkpoint intervals changed units**
   - Old: time-based (60 seconds)
   - New: epoch-based (5 epochs)
   - Could cause excessive I/O if epochs are short

5. **Worker count calculation**
   - No longer rounds to power of 2
   - Could cause suboptimal CPU/memory alignment

6. **Batch processing changes**
   - Removed dual-policy code that may have had side effects
   - Changed how tensors flow through the system

### Agent & Policy Issues
7. **Agent initialization path changed**
   - New AgentConfig system
   - Different factory function for creating agents
   - Weight initialization might be different

8. **Feature normalization changes**
   - Remapping logic was refactored
   - Could corrupt observations if mapping is wrong
   - Unknown features handled differently

9. **Network architecture defaults**
   - clip_range defaults to 0
   - analyze_weights_interval changed
   - Possible initialization differences

### Reward & Learning Issues
10. **Reward signal problems**
    - Only getting heart rewards, no intermediate progress
    - Altar 3x harder means fewer hearts collected
    - No shaped rewards to guide learning

11. **Value function bootstrap**
    - Could be learning wrong values without shaped rewards
    - GAE advantage calculation affected by sparse rewards

12. **Exploration issues**
    - With only sparse heart rewards, exploration is critical
    - Entropy coefficient might need adjustment
    - Action distribution could be affected

### Performance & System Issues
13. **PufferLib buffer management**
    - Changed to `set_buffers(env, buf)` 
    - Could add memory copy overhead

14. **Stats collection overhead**
    - Processing inline vs deferred
    - More synchronous operations

15. **Learning rate scheduling**
    - Hyperparameter scheduler changes
    - Could affect convergence

### Observation & Action Space Issues
16. **Map diversity**
    - Without blocks, environments too simple
    - Less interesting navigation challenges
    - Agents might not learn robust policies

17. **Action masking**
    - Invalid action handling might have changed
    - Could waste actions on impossible moves

18. **Observation encoding**
    - Feature IDs might be mapped differently
    - Could confuse the neural network

## Most Likely Culprits for 0.5 vs 15 Hearts

1. **Altar is 3x harder** (3 batteries vs 1) - HUGE impact
2. **No shaped rewards** - Agents don't know ore/batteries are valuable
3. **Missing blocks** - Environment too simple, different dynamics
4. **Feature remapping** - Observations might be corrupted
5. **Environment config switching** - Mid-episode config changes

## Verification Needed

1. **Confirm environment is using our recipe config**
   - Added logging to verify altar costs and rewards
   - Need to check during actual training

2. **Check if agents are even trying to collect resources**
   - Without shaped rewards, they might not learn to collect ore/batteries
   - Need to monitor intermediate resource collection

3. **Verify observation encoding**
   - Check if feature IDs are consistent
   - Ensure remapping isn't corrupting observations

## Next Steps

1. **Run with verbose logging** to confirm environment config
2. **Monitor resource collection** (ore, batteries) not just hearts
3. **Try even easier altar** (maybe 0.5 batteries?)
4. **Check if agents are exploring** or just stuck
5. **Verify feature normalization** is working correctly
6. **Test with MORE shaped rewards** to guide learning better