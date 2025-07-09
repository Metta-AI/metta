# Robust Sweep State Management

This module provides comprehensive state tracking for sweep runs to prevent runs from appearing stuck in "running" state forever.

## Overview

The `RobustSweepStateManager` class provides:
- Clear lifecycle states (initializing, running, evaluating, success, failure, etc.)
- Heartbeat mechanism to detect stuck/crashed runs
- Timeout detection
- Error tracking and recovery
- Context manager support for automatic state handling

## States

The `SweepRunState` enum defines the following states:

- **INITIALIZING**: Setting up, loading previous runs
- **RUNNING**: Training in progress
- **EVALUATING**: Evaluation in progress
- **SUCCESS**: Completed successfully
- **FAILURE**: Failed with error
- **TIMEOUT**: Timed out (no heartbeat)
- **CRASHED**: Unexpected termination
- **CANCELLED**: User cancelled

## Usage

### Basic Usage

```python
from metta.sweep.state_manager import SweepRunState, RobustSweepStateManager

# Create state manager with 30 minute timeout
state_manager = RobustSweepStateManager(wandb_run, timeout_minutes=30)

# Set initial state
state_manager.set_state(SweepRunState.INITIALIZING)

# Start training
state_manager.set_state(SweepRunState.RUNNING)

# Send periodic heartbeats during training
for epoch in range(num_epochs):
    train_epoch()
    state_manager.heartbeat()  # Keep alive signal

# Start evaluation
state_manager.set_state(SweepRunState.EVALUATING)

# Record success with metrics
state_manager.set_state(
    SweepRunState.SUCCESS,
    score=0.95,
    train_time=120.0,
    eval_time=15.0
)
```

### Context Manager Usage

```python
# Automatic state handling with context manager
with RobustSweepStateManager(wandb_run) as state_manager:
    # Automatically sets RUNNING state

    # Do training
    for epoch in range(num_epochs):
        train_epoch()
        state_manager.heartbeat()

    # Automatically sets SUCCESS on normal exit
    # or FAILURE with error details on exception
```

### Error Handling

```python
state_manager = RobustSweepStateManager(wandb_run)

try:
    state_manager.set_state(SweepRunState.RUNNING)
    train_model()
except Exception as e:
    # Automatically determines appropriate failure state
    state_manager.handle_exception(e)
    # Sets TIMEOUT for TimeoutError
    # Sets CANCELLED for KeyboardInterrupt
    # Sets FAILURE for other exceptions
```

### Timeout Detection

```python
# Create with custom timeout
state_manager = RobustSweepStateManager(wandb_run, timeout_minutes=60)

# Check if timed out
if state_manager.check_timeout():
    state_manager.set_state(
        SweepRunState.TIMEOUT,
        error="No heartbeat for 60 minutes"
    )
```

## Integration with Existing Code

### In protein_wandb.py

```python
def __init__(self, protein, wandb_run=None):
    self._wandb_run = wandb_run or wandb.run
    self._state_manager = RobustSweepStateManager(self._wandb_run)

    # Set initializing state
    self._state_manager.set_state(SweepRunState.INITIALIZING)

    # Load previous observations...

    # Set running state when ready
    self._state_manager.set_state(SweepRunState.RUNNING)
```

### In sweep_eval.py

```python
# During evaluation
state_manager = RobustSweepStateManager(wandb_run)
state_manager.set_state(SweepRunState.EVALUATING)

try:
    # Run evaluation
    results = evaluate_policy()

    # Record success
    state_manager.set_state(
        SweepRunState.SUCCESS,
        **results
    )
except Exception as e:
    state_manager.handle_exception(e)
```

### In training loop

```python
# Add heartbeats to prevent timeout
for epoch in range(num_epochs):
    # Training code...

    # Send heartbeat every epoch
    if hasattr(self, 'state_manager'):
        self.state_manager.heartbeat()
```

## Benefits

1. **Clear Run Status**: No more runs stuck showing "running" forever
2. **Debugging**: Error messages and stack traces are captured
3. **Monitoring**: Can detect and handle stuck runs programmatically
4. **Reliability**: Graceful handling of crashes and timeouts
5. **Metrics**: Runtime and other metrics are automatically tracked

## WandB Summary Fields

The state manager updates the following fields in `wandb.run.summary`:

- `protein.state`: Current state value
- `protein.last_update`: ISO timestamp of last update
- `protein.runtime_seconds`: Total runtime in seconds
- `protein.heartbeat`: ISO timestamp of last heartbeat
- `protein.error`: Error message (for failure states)
- Any additional fields passed to `set_state()`
