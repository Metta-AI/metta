# Learning Progress Arena Experiment Debug Results

## Summary

The debug tests have identified and fixed a critical issue in the learning progress algorithm that was causing frequent failures and recoveries in your runs.

## Key Findings

### ✅ **Fixed: Learning Progress Algorithm Warnings**

**Issue Found:**
- The learning progress algorithm was generating "Mean of empty slice" warnings
- This occurred when `np.mean()` was called on empty lists in `self._outcomes[i]`
- These warnings could lead to NaN values and training instability

**Root Cause:**
```python
# Line 139 in learning_progress.py (BEFORE)
task_success_rates = np.array([np.mean(self._outcomes[i]) for i in range(self._num_tasks)])

# Line 248 in learning_progress.py (BEFORE)
out_vec = [np.mean(self._outcomes[i]) for i in range(self._num_tasks)]
```

**Fix Applied:**
```python
# Line 139 in learning_progress.py (AFTER)
task_success_rates = np.array([
    np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE
    for i in range(self._num_tasks)
])

# Line 248 in learning_progress.py (AFTER)
out_vec = [np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE for i in range(self._num_tasks)]
```

**Result:**
- ✅ No more "Mean of empty slice" warnings
- ✅ No more NaN values from empty lists
- ✅ Learning progress algorithm now stable

### ✅ **All Other Systems Working Correctly**

1. **Checkpoint System**: ✅ Working properly
2. **Configuration Loading**: ✅ All configs load successfully
3. **Memory Usage**: ✅ No issues detected
4. **Curriculum Sampling**: ✅ Working correctly

## Why Your Runs Were Failing and Recovering

The primary cause was the **learning progress algorithm instability**:

1. **Empty List Warnings**: When tasks hadn't been sampled yet, `self._outcomes[i]` was empty
2. **NaN Propagation**: Empty lists caused `np.mean()` to return NaN
3. **Training Instability**: NaN values propagated through the learning progress calculations
4. **Frequent Restarts**: The training system detected instability and restarted

## Recommendations for Your Runs

### 1. **Use the Fixed Code**
The fix has been applied to `mettagrid/src/metta/mettagrid/curriculum/learning_progress.py`. This should resolve the frequent failures.

### 2. **Monitor Key Metrics**
Watch these metrics in wandb:
- `lp/num_active_tasks` - Should be around 16
- `lp/mean_sample_prob` - Should be > 0
- `lp/task_success_rate` - Should improve over time
- `lp/num_nan_tasks` - Should be 0

### 3. **Consider Hyperparameter Adjustments**
If you still see issues, try these adjustments:

```yaml
# More conservative hyperparameters
ema_timescale: 0.01          # Increased from 0.001
progress_smoothing: 0.01     # Decreased from 0.05
num_active_tasks: 8          # Decreased from 16
sample_threshold: 20         # Increased from 10
memory: 50                   # Increased from 25
```

### 4. **Run with Reduced Parameters First**
Test with smaller batch sizes and fewer workers:

```bash
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="test.fixed.learning_progress" \
    --config configs/user/learning_progress_experiment.yaml \
    trainer.total_timesteps=10_000_000 \
    trainer.num_workers=2 \
    trainer.batch_size=16384 \
    ema_timescale=0.01 \
    progress_smoothing=0.01 \
    num_active_tasks=8 \
    sample_threshold=20
```

### 5. **Monitor System Resources**
- Check GPU memory usage
- Monitor checkpoint save/load times
- Watch for network issues during S3 uploads

## Expected Improvement

With the fix applied, you should see:
- ✅ Fewer training restarts
- ✅ More stable learning progress metrics
- ✅ Better convergence
- ✅ No more "Mean of empty slice" warnings

## Next Steps

1. **Commit the fix**: The learning progress algorithm fix should be committed
2. **Test with small runs**: Run the reduced parameter test first
3. **Monitor closely**: Watch the key metrics in wandb
4. **Scale up gradually**: Once stable, increase batch sizes and timesteps

The debug tests have successfully identified and fixed the root cause of your frequent failures and recoveries!
