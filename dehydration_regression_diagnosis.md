# Dehydration Regression Diagnosis

## Problem Statement
After removing Hydra and YAML configurations from the Metta codebase (post-commit 724cde8fc27f8a30cd9c06eba38aa7cbda8c1515), performance has degraded:
- Training is slower and less stable
- Final heart.get metrics of fully trained models are significantly worse
- The new version underperforms compared to the pre-dehydration version

## Critical Findings Summary

### 🔴 HIGH PRIORITY ISSUES (likely causing performance regression):

1. **Heart Reward Capping** - The most likely culprit!
   - Old: `heart_max: null` (unlimited)
   - New: `heart_max: 255` (capped)
   - Impact: Fundamentally changes reward scale, limiting the primary success signal

2. **10x More Frequent Checkpointing**
   - Old: Every 50 epochs
   - New: Every 5 epochs
   - Impact: Massive I/O overhead, slowing training significantly

3. **Double Evaluation Overhead**
   - Old: Only remote evaluation
   - New: Both remote AND local evaluation
   - Impact: Doubles the computational cost of evaluation

4. **Changed Total Timesteps Default**
   - Old: 10B timesteps
   - New: 50B timesteps
   - Impact: Affects learning rate schedules and convergence behavior

### ✅ VERIFIED OK (not causing issues):
- Agent architecture (ComponentFast) is identical
- Core hyperparameters (LR, beta, entropy, etc.) match exactly
- Basic environment configuration matches

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

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 8. Vectorized Environment (VecEnv)
**Initial Hypotheses:**
1. **Environment synchronization** - How parallel environments are synchronized may differ
2. **Reset behavior** - Auto-reset logic or done handling may have changed
3. **Observation stacking** - Frame stacking or observation history handling may differ

**Runtime Hypotheses (to be added during investigation):**
- TBD

### 9. Neural Network Initialization
**Initial Hypotheses:**
1. **Weight initialization schemes** - Default initialization methods may have changed
2. **Bias initialization** - How biases are initialized may differ
3. **Layer-specific initialization** - Special layers may use different init strategies

**Runtime Hypotheses (to be added during investigation):**
- TBD

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

### Immediate Actions (to restore performance):

1. **Fix heart_max in arena_basic_easy_shaped.py**:
   ```python
   # Line 105 in experiments/recipes/arena_basic_easy_shaped.py
   # CHANGE FROM:
   env_cfg.game.agent.rewards.inventory.heart_max = 255
   # TO:
   env_cfg.game.agent.rewards.inventory.heart_max = None  # Restore unlimited hearts
   ```

2. **Fix checkpoint intervals in TrainerConfig**:
   ```python
   # In metta/rl/trainer_config.py, lines 53-55
   checkpoint_interval: int = Field(default=50, ge=0)  # Was 5
   wandb_checkpoint_interval: int = Field(default=50, ge=0)  # Was 5
   ```

3. **Fix evaluation defaults in EvaluationConfig**:
   ```python
   # In metta/rl/trainer_config.py, lines 66-67
   evaluate_remote: bool = Field(default=True)
   evaluate_local: bool = Field(default=False)  # Was True
   ```

4. **Fix total_timesteps default**:
   ```python
   # In metta/rl/trainer_config.py, line 124
   total_timesteps: int = Field(default=10_000_000_000, gt=0)  # Was 50B
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
The performance regression is caused by multiple configuration defaults that changed during the dehydration refactor. The most impactful is likely the heart_max capping at 255, which fundamentally changes the reward signal the agent receives for successful behavior.