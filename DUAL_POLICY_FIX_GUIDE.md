# Dual Policy Training Fix Guide

## The Problem

Your dual policy training shows flat graphs with rare spikes because:

1. **Episode Length Mismatch**: Episodes are 1000 steps long, but training happens every ~341 steps
2. **Rewards Only at Episode Completion**: Rewards are only logged when ALL 24 agents finish
3. **No Intermediate Feedback**: Without shaped rewards, there's no signal between episodes

## The Solution

### 1. Use the New Recipe with Shaped Rewards

```bash
# Test locally
python tools/run.py experiments.recipes.dual_policy_shaped:test

# Run on Skypilot
devops/skypilot/launch.py experiments.recipes.dual_policy_shaped:train run=dual_policy_shaped
```

### 2. Key Fixes Implemented

#### A. Shaped Rewards (arena_basic_easy_shaped)

- Provides continuous rewards for collecting items (ore: 0.1, battery: 0.8, laser: 0.5)
- Heart rewards scale up to 255 for maximum signal
- Easier objectives (altar only needs 1 battery instead of 3)

#### B. Reduced Episode Length

- Changed `max_steps` from 1000 to 256
- Now episodes complete ~3x per training iteration
- Better alignment with training frequency

#### C. Continuous Reward Tracking (Optional Enhancement)

If you want to implement continuous tracking in `dual_policy.py`:

1. Add to `__init__`:

```python
self._step_rewards_buffer: list[Dict[str, list[float]]] = []
self._step_count: int = 0
self._buffer_size: int = 100
```

2. Add the tracking methods from `dual_policy_continuous_tracking.py`

3. Modify `mettagrid_env.py` line 159 to pass rewards:

```python
infos.update(self.dual_policy_handler.compute_step_stats(self, self._steps, rewards))
```

### 3. Expected Results

With these fixes, you should see:

- **Smooth reward curves** instead of flat lines
- **Continuous progression** showing learning
- **Clear separation** between training and NPC performance
- **More frequent updates** to your graphs

### 4. Monitoring Tips

Watch these metrics in WandB:

- `env_dual_policy/trained/reward_mean` - Should show gradual increase
- `env_dual_policy/npc/reward_mean` - Should remain stable (fixed policy)
- `overview/reward` - Overall training progress

### 5. Additional Optimizations

If you still see issues:

- Reduce `max_steps` further to 128 or 64
- Increase evaluation frequency: `evaluate_interval=100`
- Use smaller batch size for more frequent updates
- Add more shaped reward components

## Quick Start

1. Use the new recipe file:

   ```bash
   cp experiments/recipes/dual_policy_shaped.py experiments/recipes/my_dual_policy.py
   ```

2. Adjust NPC checkpoint if needed:

   ```python
   NPC_CHECKPOINT = "your_checkpoint_here"
   ```

3. Run training:
   ```bash
   python tools/run.py experiments.recipes.my_dual_policy:train
   ```

## Why This Works

- **Shaped rewards** provide signal at every step, not just episode end
- **Shorter episodes** ensure rewards are logged frequently
- **Aligned frequencies** mean training iterations see episode completions
- **Continuous tracking** captures the actual learning signal

Your training was likely working before, but the metrics couldn't capture it. These changes make the learning process
visible!
