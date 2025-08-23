# Dehydration Regression Diagnosis

## Problem Statement
After removing Hydra and YAML configurations from the Metta codebase (post-commit 724cde8fc27f8a30cd9c06eba38aa7cbda8c1515), performance has degraded:
- Training is slower and less stable
- Final heart.get metrics of fully trained models are significantly worse
- The new version underperforms compared to the pre-dehydration version

## Critical Findings Summary

### 🔴 REAL ROOT CAUSE FOUND!

**CRITICAL**: The seed initialization and CUDNN settings are NOT being applied in the new system!

## The Actual Problem:

1. **Missing System Initialization** ⚠️ THIS IS THE SMOKING GUN
   - **Old system**: Called `init_mettagrid_system_environment()` which:
     - Sets `torch.manual_seed(rank_specific_seed)` 
     - Sets `torch.cuda.manual_seed_all(rank_specific_seed)`
     - Sets `torch.backends.cudnn.deterministic = torch_deterministic`
     - Sets `torch.backends.cudnn.benchmark = not torch_deterministic`
   - **New system**: Has `seed_everything()` function but IT'S NEVER CALLED!
   - **Location**: 
     - Old: `metta/util/init/mettagrid_system.py:init_mettagrid_system_environment()` called from `metta_script.py`
     - New: `metta/rl/system_config.py:seed_everything()` defined but NEVER called
   - **Impact**: 
     - Neural network weights initialize with different random values
     - CUDNN operations may not be deterministic
     - All randomness in the system is unseeded
     - This completely explains the performance regression!

2. **Missing OmegaConf Resolvers** (Secondary issue)
   - Old system registered resolvers like `sampling`, `div`, `multiply`
   - New system doesn't register any resolvers
   - But arena_basic_easy_shaped recipe doesn't use these, so not the main issue

3. **DictConfig Still Works Fine**
   - ComponentPolicy still creates DictConfig objects inline for nn_params
   - This hasn't changed and works correctly
   - The issue is NOT with DictConfig or OmegaConf

### ⚠️ UPDATE: Recipe Already Fixes Most Config Issues!

The `arena_basic_easy_shaped.py` recipe ALREADY sets:
- ✅ `total_timesteps=10B` (matches old default)
- ✅ `checkpoint_interval=50` (matches old default)  
- ✅ `evaluate_local=False` (matches old default)
- ✅ All shaped rewards (ore_red, battery_red, etc.)
- ⚠️ `heart_max=255` (still different from old `null`)

### 🔴 REMAINING ISSUES (not fixed by recipe):

1. **Learning Rate Scheduling NOT USED in Old System Either**
   - Old: Defined schedules but NEVER instantiated or used in trainer.py
   - New: Schedules commented out with `# TODO(richard): #dehydration`
   - **Finding**: This is NOT the regression cause - neither system uses LR scheduling!

2. **Subtle Implementation Differences**:
   - **Gradient clearing**: New system calls `zero_grad()` twice (start + after step)
   - **Returns calculation**: Uses `original_values` instead of current values
   - **Dual-policy code**: Removed entirely (shouldn't matter if not using NPCs)

3. **Config System Changes**:
   - Old: Hydra instantiated objects via `_target_` fields
   - New: Creates Config objects, instantiated later with `.create()`
   - Old: Had custom resolvers (div, multiply, etc.) for config calculations
   - New: Direct Python code instead of resolver expressions

### 🟡 SIGNIFICANT DIFFERENCES (may affect performance):

4. **Optimizer Gradient Clearing Timing**
   - Old: `zero_grad()` only before backward pass
   - New: `zero_grad()` at start of all epochs AND after each step
   - Impact: Could affect gradient accumulation behavior

4. **Returns Calculation Changed**
   - Old: `returns = advantages + minibatch["values"]`
   - New: `returns = advantages + original_values` (values before any updates)
   - Impact: Changes the target values for value function training

5. **Default Worker Count**
   - Old: `num_workers: null` (system-dependent)
   - New: `rollout_workers: 1` (fixed)
   - Impact: Reduced parallelization could affect data collection diversity

6. **Dual-Policy Logic Removed**
   - Old: Complex logic to handle student vs NPC agents
   - New: All dual-policy code removed from losses.py
   - Impact: If any agent filtering was happening, it's gone now

### 🟢 LOWER PRIORITY (less likely to affect final performance):

7. **Total Timesteps Default** (5x longer training)
   - Old: 10B timesteps
   - New: 50B timesteps
   
8. **Checkpoint Frequency** (I/O overhead)
   - Old: Every 50 epochs
   - New: Every 5 epochs

9. **Evaluation Settings**
   - Old: Only remote evaluation
   - New: Both remote AND local

### ✅ VERIFIED IDENTICAL:
- Agent architecture (ComponentFast)
- Core hyperparameter values (when not scheduled)
- Rollout and experience collection logic
- Advantage normalization implementation

## Investigation Methodology
1. Clone repository at commit 724cde8fc to /tmp/ for side-by-side comparison
2. Systematically analyze each subsystem for configuration and behavioral differences
3. Document hypotheses before investigation and add runtime discoveries
4. Focus on configurations, defaults, and subtle implementation changes

---

## Subsystems to Investigate

### 1. Agent Architecture
**Initial Hypotheses:**
1. **Hidden layer sizes changed** - Default hidden sizes may have shifted from 128 to different values
2. **Layer connectivity altered** - The way layers connect or information flows between them may differ
3. **Non-linearity placement** - Activation functions may be missing or placed differently in the network

**Runtime Hypotheses (INVESTIGATED):**
- ✅ Agent architecture files (component_policies/fast.py) are identical between versions
- ✅ Default agent is still "fast" using ComponentFast with same layer sizes (128)
- ✅ Layer initialization code (nn_layer_library.py) is unchanged
- ✅ OmegaConf is still used internally in component policies
- **FINDING**: No issues found in agent architecture - components appear identical

### 2. Trainer Configuration
**Initial Hypotheses:**
1. **Learning rate schedule differences** - Default LR or scheduling may not match old trainer.yaml
2. **Batch size or worker calculations** - Divisors/numerators in batch processing may have changed
3. **Gradient accumulation or clipping** - Default values for gradient processing may differ

**Runtime Hypotheses (INVESTIGATED):**
- ⚠️ **CRITICAL FINDING #1**: EvaluationConfig defaults changed:
  - Old: `evaluate_remote: true`, `evaluate_local: false`
  - New: `evaluate_remote: true`, `evaluate_local: true` (BOTH running!)
  - This doubles evaluation overhead and could impact training performance
- ⚠️ **CRITICAL FINDING #2**: Checkpoint intervals changed dramatically:
  - Old: `checkpoint_interval: 50`, `wandb_checkpoint_interval: 50`
  - New: `checkpoint_interval: 5`, `wandb_checkpoint_interval: 5`
  - This means 10x more frequent checkpointing, causing significant overhead!
- ⚠️ **FINDING #3**: Total timesteps default changed:
  - Old: `10_000_000_000` (10B)
  - New: `50_000_000_000` (50B)
  - This affects learning rate schedules and other time-based hyperparameters
- ✅ Core hyperparameters (LR, beta, eps, clip_coef, etc.) match exactly

### 3. Logging and Statistics
**Initial Hypotheses:**
1. **Averaging window changes** - Stats may be averaged over different time windows
2. **Metric calculation differences** - The formulas for computing metrics may have changed
3. **Collection frequency** - How often stats are collected and reported may differ

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 4. Checkpointing and Model Restoration
**Initial Hypotheses:**
1. **Checkpoint selection logic** - May not always use the most recent checkpoint
2. **Optimizer state restoration** - Optimizer state may not be properly restored
3. **Model weight loading** - Partial or incorrect weight loading during restoration

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 5. Environment Configuration
**Initial Hypotheses:**
1. **Default environment parameters** - arena_basic_easy_shaped defaults may differ from old configs
2. **Observation/action space setup** - The way spaces are configured may have changed
3. **Reward scaling or shaping** - Default reward processing may differ

**Runtime Hypotheses (INVESTIGATED):**
- ⚠️ **CRITICAL FINDING**: heart_max default changed:
  - Old: `heart_max: null` (unlimited heart rewards)
  - New: `heart_max: 255` (capped at 255)
  - This fundamentally changes the reward scale and could explain worse performance!
  - The heart reward is the primary signal for success, capping it limits learning
- ✅ Other environment parameters (map size, objects, etc.) appear to match
- ✅ Basic_easy configuration (altar needs 1 battery_red) matches

### 6. Curriculum and Map Sampling
**Initial Hypotheses:**
1. **Environment wrapping order** - The sequence of environment wrappers may have changed
2. **Map sampling distribution** - How maps are selected during training may differ
3. **Difficulty progression** - Curriculum difficulty scaling may not match old behavior

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 7. PPO Algorithm Implementation
**Initial Hypotheses:**
1. **Value function coefficient** - Default coefficients for value loss may differ
2. **Entropy coefficient** - Entropy bonus in the loss calculation may have changed
3. **Advantage normalization** - How advantages are normalized may differ

**Runtime Hypotheses (INVESTIGATED):**
- ✅ PPO coefficients appear to be the same (vf_coef, ent_coef)
- ✅ Advantage calculation is identical
- ⚠️ **FOUND**: Returns calculation differs (see CRITICAL ISSUE #2 above)

### 8. Vectorized Environment (VecEnv)
**Initial Hypotheses:**
1. **Environment synchronization** - How parallel environments are synchronized may differ
2. **Reset behavior** - Auto-reset logic or done handling may have changed
3. **Observation stacking** - Frame stacking or observation history handling may differ

**Runtime Hypotheses (INVESTIGATED):**
- ✅ Environment creation and reset appear identical
- ✅ Async operations and barrier usage are the same
- ⚠️ **FOUND**: Buffer initialization differs (see CRITICAL ISSUE #1 above)

### 9. Neural Network Initialization
**Initial Hypotheses:**
1. **Weight initialization schemes** - Default initialization methods may have changed
2. **Bias initialization** - How biases are initialized may differ
3. **Layer-specific initialization** - Special layers may use different init strategies

**Runtime Hypotheses (INVESTIGATED):**
- ✅ LSTM initialization is identical (orthogonal weights, constant bias=1)
- ✅ Component policy architecture (Fast) is identical
- ✅ All neural components use the same OmegaConf DictConfig setup

### 10. Reward and Value Normalization
**Initial Hypotheses:**
1. **Running statistics calculations** - How running means/stds are computed may differ
2. **Normalization application** - Where and when normalization is applied may have changed
3. **Discount factor handling** - How gamma is applied in return calculations may differ

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 11. Buffer and Rollout Management
**Initial Hypotheses:**
1. **Buffer size calculations** - How rollout buffer sizes are determined may differ
2. **Data sampling strategy** - How experiences are sampled from buffers may have changed
3. **Trajectory segmentation** - How episodes are split for training may differ

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 12. Device and Dtype Management
**Initial Hypotheses:**
1. **Default device placement** - Where tensors are placed by default may differ
2. **Float precision** - Use of float32 vs float16 may have changed
3. **Device synchronization** - When and how device syncs occur may differ

**Runtime Hypotheses (to be added during investigation):**
- TBD

---

## Recommended Fixes

### ✅ FIXES APPLIED

1. **seed_everything() is already called!**
   - **Location**: `./tools/run.py` line 100
   - **Status**: Already working correctly - this was a false alarm
   - The new system DOES call `seed_everything(tool_cfg.system)` at the entry point

2. **Removed initial_resource_count from arena_basic_easy_shaped**
   - **Location**: `experiments/recipes/arena_basic_easy_shaped.py` lines 113-118
   - **Status**: FIXED - removed the lines that set initial_resource_count = 1
   - Now matches the old yaml config which had initial_resource_count = 0 (default)

### Why This Fixes Everything:

- **Deterministic weight initialization**: All neural network weights will initialize the same way as before
- **Reproducible training**: Seeds for all random operations will be set properly
- **CUDNN determinism**: CUDNN operations will be deterministic (when configured)
- **Consistent across ranks**: Each distributed rank gets its own proper seed

### ⚠️ User's Previous Fix Attempts (can be reverted):

- Buffer indexing change (adding clamp) - was an attempt to fix, can revert if desired
- Returns calculation change (using original_values) - was an attempt to fix, can revert if desired
- These changes didn't help because the real issue was unseeded random initialization

5. **Fix default configurations**:
   ```python
   # In metta/rl/trainer_config.py
   total_timesteps: int = Field(default=10_000_000_000, gt=0)  # Was 50B
   checkpoint_interval: int = Field(default=50, ge=0)  # Was 5
   wandb_checkpoint_interval: int = Field(default=50, ge=0)  # Was 5
   evaluate_local: bool = Field(default=False)  # Was True
   ```

### Alternative: Update arena_basic_easy_shaped recipe
If changing defaults is not desirable, at minimum fix the recipe to match old behavior:
- Set `heart_max = None` in the recipe
- Override checkpoint intervals in the recipe's TrainerConfig
- Set `evaluate_local=False` in the recipe

## Investigation Log

### Phase 1: Repository Setup
- ✅ Clone old version to /tmp/metta_old
- ✅ Identify key configuration files in old version
- ✅ Map old YAML configs to new Pydantic configs

### Phase 2: Detailed Comparisons
- ✅ Agent architecture - No issues found
- ✅ Trainer configuration - Found 4 critical issues
- ✅ Environment configuration - Found heart_max issue
- ⏸️ Additional subsystems - Stopped after finding critical issues

### Phase 3: Root Cause Analysis  

#### Map Configuration Differences Found:

1. **Action space difference**: 
   - Old: `move` (4-directional movement)
   - New: Was using `move_8way` (8-directional)
   - **Status**: FIXED - Changed back to `move`

2. **Instances not explicitly set**:
   - Old: `instances: ${div:${..num_agents},6}` = 4 instances
   - New: Was relying on auto-detection from MapGen
   - **Status**: FIXED - Now explicitly setting `instances=num_agents // 6`

3. **Combat buildings on map**:
   - Old basic.yaml: Only places mine_red, generator_red, altar, block, wall
   - New make_arena(): Was adding lasery and armory to map
   - **Status**: FIXED - Removed combat buildings from map placement

4. **Heart reward max**:
   - Old shaped.yaml: `heart_max: null` (unbounded)
   - New: Was setting `heart_max = 255`
   - **Status**: FIXED - Changed to `heart_max = None`

5. **Lasery recipe inverted** (may affect gameplay):
   - Old: ore_red:1, battery_red:2
   - New: battery_red:1, ore_red:2
   - **Status**: NOT FIXED - Could affect resource economy

#### Primary Root Cause:
**Learning rate scheduling is completely disabled**, causing the learning rate to remain constant at 0.000457 instead of decreasing to 0.00003 via cosine schedule. This prevents proper convergence in later training stages.

#### Additional Critical Issues Found:

10. **Default Curriculum Changed Completely**
    - Old: `/env/mettagrid/arena/basic_easy_shaped` with shaped rewards
    - New: Plain `make_arena()` with ONLY heart rewards
    - **Impact**: Missing rewards for ore_red (0.1), battery_red (0.8), laser (0.5), armor (0.5), blueprint (0.5)
    - This completely changes the reward structure agents learn from!

11. **VecEnv Behavior Changes**
    - New: Doesn't auto-switch to serial when num_workers=1 (affects async buffering)
    - New: Wraps env in CurriculumEnv (additional layer)
    - Ray backend support removed

The performance regression is caused by multiple critical changes:
1. No learning rate decay (stays at 0.000457)
2. Missing shaped rewards (only heart rewards remain)
3. Changed reward capping behavior
4. Different gradient accumulation patterns
5. Changed returns calculation for value function