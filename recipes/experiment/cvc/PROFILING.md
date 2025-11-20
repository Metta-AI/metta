# Profiling Thinky Supervised to PPO

## Quick Start on GPU Machine

### Method 1: Use Profiling Script (Recommended)

```bash
# SSH to GPU machine
ssh your-gpu-machine

# Run profiling (10k steps with detailed logs)
cd /path/to/metta3
uv run python recipes/experiment/cvc/profile_thinky_supervised.py
```

This will:
- Run with detailed timing breakdowns logged every 10 rollouts
- Generate PyTorch profiler trace files
- Show you exactly where time is spent

### Method 2: Use Built-in Profiler with Main Recipe

```bash
uv run ./tools/run.py cvc.thinky_supervised_to_ppo \
    run=msb_profile \
    base_missions=extractor_hub_30 \
    supervised_steps=5_000 \
    trainer.total_timesteps=10_000 \
    trainer.profiler.enabled=True \
    trainer.profiler.wait_steps=2 \
    trainer.profiler.warmup_steps=2 \
    trainer.profiler.active_steps=4 \
    trainer.profiler.record_shapes=True \
    evaluator.evaluate_local=False  # Disable eval for profiling
```

### Method 3: Add Custom Timing Logs

In your recipe, change the import to use the profiled version:

```python
# Instead of:
# from metta.rl.loss.action_supervised_and_critic import ActionSupervisedAndCriticConfig

# Use:
from metta.rl.loss.action_supervised_and_critic_profiled import ActionSupervisedAndCriticConfig
```

## Reading Results

### Timing Logs

Look for output like this in your logs:

```
============================================================
Rollout Timing Profile (agent_step=5120)
============================================================
  Rollout Total:         15.23 ms
    - Student Forward:    3.45 ms
    - Teacher Total:     10.12 ms
        • Reshape:        0.15 ms
        • GPU->CPU:       5.23 ms  <-- BOTTLENECK if high
        • Nim Calls:      3.89 ms
        • Stack+Back:     0.85 ms
    - Replay Store:       1.66 ms
============================================================
```

### PyTorch Profiler Trace

1. Find the trace file:
   ```bash
   ls ./train_dir/msb_profile_*/profiler_trace_*.json
   ```

2. Download to your local machine:
   ```bash
   scp your-gpu-machine:/path/to/profiler_trace_*.json .
   ```

3. Open in Chrome:
   - Go to `chrome://tracing`
   - Click "Load" and select the trace file
   - Look for operations taking the most time

## Common Bottlenecks

### 1. GPU -> CPU Transfer (~50% of time)
**Symptom**: `GPU->CPU` time is 5+ ms
**Fix**: Pre-allocate CPU buffers, reduce transfer frequency

### 2. Nim Agent Calls (~30% of time)
**Symptom**: `Nim Calls` time is 3+ ms per environment
**Fix**: This is C++ code, hard to optimize. Consider reducing num_envs

### 3. Replay Buffer Store (~10% of time)
**Symptom**: `Replay Store` time is 2+ ms
**Fix**: Already optimized, likely not the bottleneck

### 4. Student Forward Pass (~10% of time)
**Symptom**: `Student Forward` time is high
**Fix**: Use mixed precision, check batch size

## Optimization Checklist

After profiling, try these optimizations in order:

1. **Reduce eval frequency** (if eval is enabled):
   ```python
   evaluator_cfg = EvaluatorConfig(
       epoch_interval=50,  # Was 10
   )
   ```

2. **Pre-allocate NumPy buffers**:
   See optimization #1 in my previous message

3. **Increase batch size** (if GPU has memory):
   ```bash
   trainer.batch_size=1024  # Try doubling
   ```

4. **Disable teacher during PPO phase**:
   Already handled by scheduler, verify it's working

5. **Use mixed precision**:
   ```bash
   trainer.mixed_precision=True
   ```

## Expected Performance

| Configuration | SPS | Notes |
|---|---|---|
| CPU (MacBook) | 150-250 | Current baseline |
| GPU (V100) supervised | 800-1200 | With teacher |
| GPU (V100) PPO | 2000-3000 | Without teacher |
| GPU (A100) supervised | 1500-2500 | With teacher |
| GPU (A100) PPO | 4000-6000 | Without teacher |

If you're getting <50% of expected SPS, there's likely a configuration issue.

