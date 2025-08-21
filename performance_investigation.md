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

## NEW FINDINGS FROM DEEP DIVE

### Critical Changes Found
1. **Curriculum system completely rewritten**
   - Old: Used Hydra-based curriculum with specific task completion tracking
   - New: Simplified curriculum that may not be sampling tasks correctly
   - Task completion uses `get_episode_rewards().mean()` which might be wrong

2. **Initial resource counts were ALWAYS 0 in old configs**
   - CORRECTION: Old configs also had `initial_resource_count: 0`
   - This was NOT the issue - setting to 1 was incorrect
   - Buildings start empty and must wait for cooldown before producing

3. **Resource production timeline issue**
   - With initial_resource_count=0 and cooldowns:
     - Mine produces first ore at step 50
     - Generator needs 25 more steps to make battery
     - Altar needs 10 more steps to make heart
   - With easy altar (1 battery): ~89 steps minimum to first heart
   - With default altar (3 batteries): ~235 steps minimum to first heart
   - Episode max_steps=1000 gives limited time to learn this sequence

4. **Shaped rewards ARE being configured correctly**
   - Logging confirms: ore_red=0.1, battery_red=0.8
   - Altar correctly set to need only 1 battery
   - But agents still stuck at 0.5 hearts

## CRITICAL ROOT CAUSE IDENTIFIED

### The desync_episodes Implementation Changed!

**THE MAIN ISSUE**: `desync_episodes` behavior changed between old and new systems!

**Old Implementation** (pre-dehydration):
- Changed `max_steps` for the FIRST episode only at environment creation
- Episode ran normally but with a different max_steps limit
- Agents still got full episodes, just of varying lengths

**New Implementation** (post-dehydration):
- Truncates the FIRST episode at a random step between 1 and max_steps
- Uses `self._early_reset` to force truncation mid-episode
- If truncation happens at step 1-50: agents get NO rewards (ore doesn't spawn until step 50)
- If truncation happens at step 50-89: agents might get 0.1 ore reward but no hearts

**Why this matters for Arena but not Navigation**:
- Navigation: Altars start with hearts ready (initial_resource_count=1), so even short episodes give rewards
- Arena: Complex resource chain takes 89+ steps minimum to get first heart
- Arena agents frequently get truncated episodes with zero or minimal rewards

**FIX APPLIED**: Set `env_cfg.desync_episodes = False` in arena_easy_shaped.py

### Secondary Issues Found

1. **CurriculumEnv wrapper now applied to ALL environments**
   - Old: MettaGridEnv created directly with curriculum
   - New: MettaGridEnv wrapped with CurriculumEnv in vecenv
   - Wrapper calls `set_env_config` after EVERY episode
   - This rebuilds map_builder and could reset agent state

2. **Checkpoint intervals changed**
   - Old: Time-based (60 seconds)
   - New: Epoch-based (5 epochs)
   - Could cause more frequent I/O

3. **PPO/Training changes**
   - Removed dual-policy training code
   - Learning rate slightly changed: 0.000457... → 0.000457
   - evaluate_interval: 300 → 50
   - No changes to PPO core algorithm, advantages, or fast.py network

4. **Performance overhead**
   - CurriculumEnv wrapper adds indirection
   - set_env_config rebuilds every episode
   - Affecting SPS (~350k vs ~450k)

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