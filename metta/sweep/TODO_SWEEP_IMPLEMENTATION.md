# TODO: Simple Sequential Sweep Implementation

## Goal
Implement a fully operational simple sequential sweep by end of day tomorrow that can run training jobs one at a time, evaluate them, and use a sequential scheduler that always schedules exactly one job at a time.

## High Priority Tasks

### 1. Update JobDefinition (First thing)
- [x] Add resource requirements to JobDefinition
  ```python
  @dataclass
  class JobDefinition:
      # ... existing fields ...
      gpus: int = 0  # Number of GPUs required
      nodes: int = 1  # Number of nodes required
      # ... rest ...
  ```

### 2. Implement Concrete Classes (Morning)
- [x] **SequentialScheduler** - Always schedules one job at a time
  - `initialize()` - Create first job with base configuration
  - `schedule()` - Handles both training and evaluation jobs in single method
  - ~~`schedule_evaluations()`~~ - DEPRECATED, logic moved to main schedule()
  - `should_stop_job()` - Return False (no early stopping for v1)
  - Keep track of trial count internally
  
- [ ] **InMemoryStore** - Simple in-memory implementation of Store protocol
  - All methods from Store protocol
  - Use dictionaries to store jobs, results, and metadata
  - File-based persistence using pickle/json for crash recovery
  
- [ ] **GridSearchOptimizer** - Provides hyperparameter configurations
  - `suggest()` - Return next configuration from grid
  - `update()` - Track which configurations have been tried
  - Keep internal state of hyperparameter grid

### 3. Integration with Existing Code (Afternoon)
- [ ] Create **SweepOrchestratorTool** in `metta/tools/sweep_orchestrator.py`
  - Similar structure to existing SweepTool
  - Use the new orchestrator instead of protein-based sweep
  - Accept recipe name (e.g., "experiments.recipes.arena")
  - Pass GPU/node requirements through to jobs
  
- [ ] Update LocalDispatcher to properly construct commands
  - Handle both train and eval job types
  - For eval jobs: use `.evaluate` instead of `.train_shaped`
  - Pass policy_uri from parent training job
  - Log GPU/node requirements (even if not enforced locally)

### 4. Testing & Debugging (Late Afternoon)
- [ ] Create minimal test script `test_sweep_orchestrator.py`
  - Test with 2-3 hyperparameter combinations
  - Use arena recipe
  - Verify jobs complete and evaluations run
  - Test with different GPU requirements
  
- [ ] Add logging throughout to track:
  - Job submission with resource requirements
  - State transitions
  - Store operations
  
- [ ] Handle edge cases:
  - Failed jobs
  - Keyboard interrupts
  - Restart after crash

## Medium Priority Tasks

### 5. Persistence & Recovery
- [ ] Implement Store persistence to disk
  - Save state after each update
  - Load state on startup if file exists
  - Allow resuming interrupted sweeps

### 6. Recipe Integration  
- [ ] Parse recipe module to extract:
  - Available hyperparameters from train functions
  - Default evaluation configurations
  - Policy URI patterns
  - Resource requirements (GPUs, nodes)

## Low Priority (If Time Permits)

### 7. Monitoring & Visualization
- [ ] Basic CLI output showing:
  - Current phase
  - Jobs completed/failed
  - Resource utilization (GPUs requested)
  - Best performing configuration so far
  
### 8. Documentation
- [ ] Usage example in README
- [ ] Docstrings for main classes

## Implementation Notes

### Key Design Decisions
1. **Sequential scheduler** - Always schedules exactly one job (no parallelism)
2. **Grid search optimizer** - Provides hyperparameter configurations systematically
3. **Local dispatch only** - Focus on subprocess execution
4. **Resource tracking** - Add GPU/node requirements even if not enforced locally
5. **In-memory store** - With optional file persistence

### File Structure
```
metta/sweep/
  sweep_orchestrator.py         # Already done (needs JobDefinition update)
  sequential_scheduler.py       # TODO: Always schedules one job
  simple_store.py              # TODO: In-memory store
  grid_search_optimizer.py     # TODO: Grid search configurations
  
metta/tools/
  sweep_orchestrator.py        # TODO: Tool wrapper

examples/
  test_sweep_orchestrator.py  # TODO: Test script
```

### Example Usage (Target)
```python
# In test script
from metta.tools.sweep_orchestrator import SweepOrchestratorTool

tool = SweepOrchestratorTool(
    sweep_name="test_sweep_1",
    recipe="experiments.recipes.arena",
    hyperparameters={
        "rewards": [True, False],
        "converters": [True, False],
    },
    max_trials=4,
    gpus_per_job=1,  # Request 1 GPU per job
    nodes_per_job=1,  # Single node jobs
)

tool.invoke()
```

## Architecture Clarification

The separation of concerns:
- **SequentialScheduler**: Controls WHEN jobs run (one at a time, always)
- **GridSearchOptimizer**: Controls WHAT configurations to try (hyperparameter grid)
- **Store**: Tracks everything (jobs, results, state)
- **Dispatcher**: HOW jobs run (subprocess with resource requirements)

The flow:
1. Scheduler asks "should we schedule a job?" → Always yes if no jobs running
2. Optimizer provides next configuration from grid
3. Job created with config + resource requirements
4. Dispatcher runs job (logging GPU/node requirements)
5. When complete, schedule evaluation
6. Repeat until all configurations tried

## Success Criteria
- [ ] Can run a sweep with 4 hyperparameter combinations sequentially
- [ ] Each job specifies GPU/node requirements (even if not enforced)
- [ ] Each training job completes and produces a policy
- [ ] Each policy gets evaluated automatically
- [ ] Results are logged and stored
- [ ] Can resume after interruption
- [ ] Clear logging shows progress and resource requirements

## Time Estimate
- First 30 mins: Update JobDefinition with GPU/node fields
- Morning (3 hours): Implement concrete classes
- Early afternoon (2 hours): Integration with tools
- Late afternoon (2-3 hours): Testing and debugging
- Evening (if needed): Polish and documentation

Total: 7-8 hours of focused work
