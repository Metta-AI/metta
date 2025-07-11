# 🐛 Bug Report: WandB sweep runs stuck in "running" state forever

## Summary
Sweep runs in WandB show as perpetually "running" even after successful completion, causing confusion in sweep monitoring and potentially affecting Protein's observation loading logic. Completed runs should transition to proper terminal states ("success", "failed", "crashed").

## Problem Details

### Current Broken Behavior
1. **Training completes successfully**: Policy saved, evaluation runs, results logged
2. **WandB state remains**: `protein.state: "running"` indefinitely
3. **UI impact**: WandB dashboard shows green "running" indicators forever
4. **Protein impact**: May skip loading observations from "running" runs in some cases
5. **Monitoring confusion**: Hard to distinguish truly active runs from completed ones

### Observable Symptoms
- WandB sweep dashboard shows all historical runs as "running"
- No clear visual indication of completion status
- Potential performance impact from querying stale "running" runs
- Misleading metrics about sweep progress and resource utilization

### Code Location
**Primary**: `metta/metta/sweep/protein_wandb.py`
**Lines**: ~230-240 (validation logic)
**Related**: `tools/train.py`, `tools/sweep_eval.py`

### Current State Management
```python
# In WandbProtein.__init__
self._wandb_run.summary.update({"protein.state": "running"})  # Set but never updated

# In _validate_run()
if run.summary.get("protein.state") == "running":
    # Checks heartbeat, but doesn't update state
    if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
        self._defunct += 1  # Counts as defunct but state stays "running"
```

## Expected Behavior
Runs should transition through proper state lifecycle:
1. **"initializing"** → during startup
2. **"running"** → during active training/evaluation
3. **"success"** → after successful completion with recorded observation
4. **"failed"** → after errors or crashes
5. **"timeout"** → after exceeding time limits

## Root Cause Analysis

### Missing State Transitions
The system sets `protein.state: "running"` but never updates it to terminal states:

1. **No success transition**: `train.py` completes but doesn't mark success
2. **No failure handling**: Crashes leave runs in "running" state
3. **No timeout handling**: Long runs aren't marked as timed out
4. **Heartbeat detection**: Identifies defunct runs but doesn't update their state

### Process Flow Gaps
```
Training Process:
├── sweep_init.py: Sets "running" ✅
├── train.py: Training happens ❓ (no state update)
├── sweep_eval.py: Evaluation happens ❓ (no state update)
└── Completion: ❌ (state never updated)
```

## Proposed Fixes

### Option 1: Add Success/Failure Transitions
```python
# In train.py or sweep_eval.py after successful completion
if wandb.run and hasattr(wandb.run, 'summary'):
    wandb.run.summary.update({
        "protein.state": "success",
        "protein.completion_time": datetime.now(timezone.utc).isoformat()
    })

# In exception handlers
except Exception as e:
    if wandb.run and hasattr(wandb.run, 'summary'):
        wandb.run.summary.update({
            "protein.state": "failed",
            "protein.error": str(e),
            "protein.failure_time": datetime.now(timezone.utc).isoformat()
        })
    raise
```

### Option 2: Retroactive State Cleanup
```python
# In protein_wandb.py _validate_run()
if run.summary.get("protein.state") == "running":
    # Check if run actually completed by looking for final metrics
    if run.summary.get("protein.objective") is not None:
        # Has final observation, should be marked as success
        run.summary.update({"protein.state": "success"})
        return True
    elif (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
        # Mark truly defunct runs
        run.summary.update({"protein.state": "timeout"})
        return False
```

### Option 3: Lifecycle Management Class
```python
class ProteinStateManager:
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run

    def set_running(self):
        self.wandb_run.summary.update({"protein.state": "running"})

    def set_success(self, objective, cost):
        self.wandb_run.summary.update({
            "protein.state": "success",
            "protein.objective": objective,
            "protein.cost": cost,
            "protein.completion_time": datetime.now(timezone.utc).isoformat()
        })

    def set_failed(self, error_msg):
        self.wandb_run.summary.update({
            "protein.state": "failed",
            "protein.error": error_msg,
            "protein.failure_time": datetime.now(timezone.utc).isoformat()
        })
```

## Impact Assessment

### Current Impact
- **UI/UX**: Confusing sweep monitoring dashboards
- **Performance**: Unnecessary API calls to query completed runs
- **Reliability**: Potential edge cases in observation loading
- **Debugging**: Hard to distinguish active vs completed experiments

### Risk Level
- **Medium-High**: Affects sweep usability and monitoring
- **Silent degradation**: No errors thrown, just poor UX
- **Scaling concern**: Gets worse with more sweep runs

## Reproduction Steps
1. Start any sweep: `./devops/sweep.sh run=test.sweep`
2. Wait for training to complete successfully
3. Check WandB dashboard - run shows as "running"
4. Start new sweep runs - old "running" entries persist
5. Over time, dashboard fills with zombie "running" entries

## Verification
After fix implementation:
1. Run should show "success" after completion
2. Failed runs should show "failed" state
3. Timed-out runs should show "timeout" state
4. WandB dashboard should clearly show run completion status
5. Protein should reliably load observations from completed runs

---
**Priority**: Medium-High - degrades sweep monitoring and user experience
