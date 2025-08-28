# Sweep Orchestration Architecture

## Overview

The sweep orchestration system is designed as a **stateless, synchronous architecture** where all state is stored in a single source of truth (the Store). This design allows for fault tolerance, easy debugging, and simple reasoning about system behavior.

## Core Design Principles

1. **Stateless Controller**: The SweepController maintains no state between iterations. It can be killed and restarted at any point without losing progress.

2. **Single Source of Truth**: All state is stored in the Store, which provides a standardized interface through RunInfo objects.

3. **Synchronous Operations**: All operations are synchronous with built-in retry logic, avoiding the complexity of async/await patterns.

4. **Protocol-Based Design**: Core components (Scheduler, Store, Dispatcher, Optimizer) are defined as Protocols, allowing for flexible implementations.

5. **Separation of Concerns**:
   - **Scheduler**: Decides WHEN jobs run and manages job lifecycle
   - **Optimizer**: Decides WHAT configurations to try
   - **Dispatcher**: Handles HOW jobs are executed
   - **Store**: Maintains all state (the source of TRUTH)

## Architecture Components

### Data Models

```python
JobDefinition     # Defines a job to be executed
RunInfo          # Standardized run information from Store
JobStatus        # Enum of possible job states
JobTypes         # LAUNCH_TRAINING, LAUNCH_EVAL
Observation      # Results from completed runs
SweepMetadata    # Aggregate statistics about the sweep
```

### Core Protocols

```python
Scheduler        # Decides which jobs to run based on all runs
Store           # Single source of truth for all state
Dispatcher      # Handles job execution mechanics
Optimizer       # Suggests hyperparameter configurations
```

### Control Flow

The main control loop in SweepController follows these steps:

1. **Fetch all runs** from Store → `list[RunInfo]`
2. **Compute metadata** from runs → `SweepMetadata`
3. **Schedule jobs** via Scheduler (handles both training and eval) → `list[JobDefinition]`
4. **Execute decisions** via Dispatcher
5. **Update transient states** and mark completions
6. **Sleep** and repeat

### State Management

RunInfo provides a computed status property based on training/eval progress:
- `PENDING`: Not started training
- `IN_TRAINING`: Training in progress
- `TRAINING_DONE_NO_EVAL`: Training complete, awaiting evaluation
- `IN_EVAL`: Evaluation in progress
- `EVAL_DONE_NOT_COMPLETED`: Evaluation done but no observation recorded
- `COMPLETED`: Fully complete with observation

## Key Design Decisions

1. **No parent_job_id**: Evaluation jobs are linked to training runs via run_id naming convention (e.g., `{run_id}_eval`)

2. **Dispatcher Selection**: Dispatchers are passed directly to the controller, no dispatch_type enum needed

3. **Resource Limits**: Controller enforces `max_parallel_jobs` to prevent resource overrun

4. **Subprocess Safety**: LocalDispatcher uses `subprocess.DEVNULL` to avoid deadlock on large outputs

5. **Retry Logic**: All Store operations wrapped with exponential backoff retry decorator

---

# TODO Lists

## 1. TODOs for First Working Sweep

### High Priority - Core Components
- [ ] **Implement WandbStore** - Concrete Store implementation using WandB
  - `fetch_runs()` - Query WandB for runs
  - `init_run()` - Create new WandB run
  - `update_run_summary()` - Update run metrics
  - `fetch_sweep_metadata()` - Get or create sweep metadata
  - Map WandB run states to JobStatus enum
  - Handle WandB API errors gracefully

- [ ] **Implement GridSearchOptimizer** - Simple hyperparameter grid search
  - Track which configurations have been tried
  - `suggest()` - Return next configuration from grid
  - Support for discrete and continuous parameters
  - Configuration deduplication

- [ ] **Fix SequentialScheduler** - Update for new architecture
  - Remove references to parent_job_id
  - Properly track evaluation status
  - Use run_id naming convention for eval jobs
  - Handle policy_uri correctly for evaluations

### High Priority - Integration
- [ ] **Create SweepOrchestratorTool** in `metta/tools/sweep_orchestrator.py`
  - Similar structure to existing tools
  - Accept recipe name and hyperparameter grid
  - Initialize Store, Scheduler, Optimizer, Dispatcher
  - Start the control loop

- [ ] **Add SweepConfig class**
  - Define protein metrics configuration
  - Hyperparameter search space
  - Resource requirements (GPUs, nodes)
  - Evaluation settings

### Testing & Validation
- [ ] **Create test script** `test_sweep_orchestrator.py`
  - Test with 2-3 hyperparameter combinations
  - Use arena recipe
  - Verify jobs complete and evaluations run
  - Test recovery after interruption

- [ ] **Add comprehensive logging**
  - Job submission and completion
  - State transitions
  - Store operations
  - Scheduler decisions

## 2. TODOs for Sweep System Improvements

### Scheduler Enhancements
- [ ] **Implement ASHAScheduler** - Successive halving for early stopping
  - Track job progress over time
  - Implement promotion/demotion logic
  - Support for different reduction factors
  - Configurable grace period

- [ ] **Implement PBTScheduler** - Population-based training
  - Track population of concurrent runs
  - Implement exploit/explore strategies
  - Support for weight inheritance
  - Checkpoint management

- [ ] **Implement BayesianScheduler** - Bayesian optimization
  - Gaussian process or tree-based surrogate models
  - Acquisition function implementation
  - Warm starting from prior runs

### Store Improvements
- [ ] **Add CachedStore wrapper** - Reduce API calls
  - In-memory cache with TTL
  - Write-through for updates
  - Invalidation strategy
  - Metrics on cache hit rate

- [ ] **Implement FileStore** - Local file-based store for testing
  - JSON/pickle serialization
  - Directory structure for organization
  - Atomic writes for consistency

- [ ] **Add Store migrations** - Handle schema evolution
  - Version tracking
  - Forward migration scripts
  - Backward compatibility

### Dispatcher Enhancements
- [ ] **Implement SkypilotDispatcher** - Cloud job dispatch
  - Sky job submission
  - Status monitoring
  - Log aggregation
  - Spot instance handling

- [ ] **Implement SlurmDispatcher** - HPC cluster dispatch
  - SLURM job submission
  - Queue management
  - Resource allocation
  - Module loading

- [ ] **Add DispatcherPool** - Multiple dispatcher types
  - Route jobs to appropriate dispatcher
  - Load balancing
  - Fallback strategies

### Optimizer Improvements
- [ ] **Add hyperparameter importance analysis**
  - Track parameter sensitivity
  - Visualization of search space
  - Pruning of unimportant parameters

- [ ] **Implement multi-objective optimization**
  - Pareto frontier tracking
  - Scalarization strategies
  - Visualization tools

### Monitoring & Observability
- [ ] **Add Prometheus metrics**
  - Job throughput
  - Resource utilization
  - Scheduler efficiency
  - Store latency

- [ ] **Create dashboard** - Real-time sweep monitoring
  - Job status visualization
  - Hyperparameter performance heatmap
  - Resource usage graphs
  - ETA predictions

- [ ] **Add sweep checkpointing**
  - Periodic state snapshots
  - Quick recovery on failure
  - Sweep migration between controllers

## 3. TODOs for Tool Methodology Integration

### Recipe Integration
- [ ] **Update recipe functions** to support sweep mode
  - Add sweep-specific configuration options
  - Support for hyperparameter injection
  - Standardized metric reporting
  - Checkpoint management

- [ ] **Create sweep-specific recipes**
  - `experiments.recipes.arena.sweep_train`
  - Simplified configuration for sweep trials
  - Automatic metric extraction
  - Built-in evaluation triggers

### Tool Framework Integration
- [ ] **Extend Tool base class** for sweep support
  - Add `sweep_mode` flag
  - Hyperparameter override mechanism
  - Metric collection hooks
  - Progress reporting

- [ ] **Create SweepTool wrapper**
  - Wrap any Tool for sweep execution
  - Automatic metric extraction
  - Resource requirement detection
  - Failure handling

### Configuration System
- [ ] **Integrate with Pydantic configs**
  - SweepConfig as Pydantic model
  - Validation of hyperparameter ranges
  - Type-safe configuration merging
  - YAML/JSON serialization

- [ ] **Add sweep section to recipes**
  - Default hyperparameter ranges
  - Suggested scheduler configurations
  - Resource recommendations
  - Metric definitions

### Workflow Integration
- [ ] **Add sweep commands to CLI**
  - `metta sweep start <recipe> --config sweep.yaml`
  - `metta sweep status <sweep_id>`
  - `metta sweep stop <sweep_id>`
  - `metta sweep analyze <sweep_id>`

- [ ] **Integrate with existing tools**
  - Auto-generate sweep configs from recipes
  - Convert existing training runs to sweep trials
  - Import/export sweep results

### Analysis Tools
- [ ] **Create sweep analysis tool**
  - Best hyperparameter extraction
  - Performance visualization
  - Statistical significance testing
  - Report generation

- [ ] **Add comparison tools**
  - Compare multiple sweeps
  - A/B testing framework
  - Regression detection
  - Performance benchmarking

### Documentation & Examples
- [ ] **Create sweep tutorial**
  - Step-by-step guide
  - Common patterns
  - Troubleshooting guide
  - Performance tips

- [ ] **Add example sweeps**
  - Grid search example
  - Bayesian optimization example
  - Population-based training example
  - Multi-objective example

### Testing Infrastructure
- [ ] **Add sweep-specific tests**
  - Unit tests for each component
  - Integration tests with mock Store
  - End-to-end tests with real recipes
  - Performance benchmarks

- [ ] **Create sweep CI/CD pipeline**
  - Automated testing of sweep changes
  - Performance regression detection
  - Resource usage monitoring
  - Compatibility testing