# Adaptive Experiments Framework

## What is the Adaptive Experiments Framework?

The adaptive experiments framework is a series of modules and protocols aiming to simplify the development, scaling, and execution of long-running experiments. It provides a clean, protocol-based architecture for orchestrating complex experimental workflows where decisions are made dynamically based on historical data and experiment state.

The framework can be used in two ways:
1. **Through specialized tools** (like `SweepTool` for Bayesian optimization) that provide purpose-built interfaces for specific experiment types
2. **Directly via `AdaptiveController`** for custom experiments that don't fit existing tool patterns

## When Should I Use Adaptive Experiments?

Adaptive experiments shine when your experiment needs:

1. **Many runs and/or decisions throughout an experiment** - Managing dozens or hundreds of training runs with complex dependencies
2. **Historical data in-experiment to inform decisions** - Using results from completed runs to guide future runs

The canonical adaptive experiment is, of course, **hyperparameter sweeps**, which is where this architecture originated from. However, the logic is flexible and lends itself to a vast array of experiment types:

- **Validating runs**: You may choose to write an experiment which picks out the best N runs of a group of experiments, and re-runs the training over different seeds for validation. Additionally, you may wish to cancel remaining seed runs if you notice that performance across the first K seeds is actually terrible.
- **Neural Architecture Search**: This becomes possible to write and test over 100s of runs as an adaptive experiment. Simply write a scheduler which produces different trainer configs.
- **Experiment batch with early termination** *(Not yet supported)*: You may launch a bunch of runs simultaneously, monitor their performance, and decide to kill aforementioned jobs if their performance is sub-par.

## How Do I Write an Adaptive Experiment?

There are two approaches depending on your needs:

### Option 1: Use a Specialized Tool (Recommended for Common Patterns)

For common experiment patterns like hyperparameter sweeps, use an existing specialized tool:

```python
# Example: Using SweepTool for Bayesian optimization
from metta.tools.sweep import SweepTool
from metta.sweep.protein_config import ProteinConfig, ParameterConfig

tool = SweepTool(
    protein_config=ProteinConfig(
        metric="evaluator/eval_arena/score",
        goal="maximize",
        parameters={
            "trainer.learning_rate": ParameterConfig(
                min=1e-5, max=1e-3,
                distribution="log_normal",
                mean=1e-4, scale="auto"
            )
        }
    ),
    max_trials=10,
    batch_size=4
)
tool.invoke({})
```

### Option 2: Use AdaptiveController Directly (For Custom Experiments)

For experiments that don't fit existing tools, use the AdaptiveController directly:

#### Step 1: Write a Scheduler

The scheduler is the brain of your experiment. It implements the `ExperimentScheduler` protocol with two required methods:

```python
from metta.adaptive.protocols import ExperimentScheduler
from metta.adaptive.models import JobDefinition, JobTypes, RunInfo, JobStatus

class MyScheduler(ExperimentScheduler):
    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """
        Decide which jobs to dispatch next based on current run state and available resources.

        Args:
            runs: All runs in the experiment (completed, running, failed)
            available_training_slots: How many LAUNCH_TRAINING jobs can be dispatched right now

        Returns:
            Jobs to dispatch. LAUNCH_TRAINING jobs should not exceed available_training_slots.
            LAUNCH_EVAL jobs don't count against the limit and can always be dispatched.
        """
        jobs = []

        # Schedule evaluation jobs for completed training
        for run in runs:
            if run.status == JobStatus.TRAINING_DONE_NO_EVAL:
                jobs.append(JobDefinition(
                    run_id=run.run_id,
                    cmd="experiments.recipes.arena.evaluate",
                    type=JobTypes.LAUNCH_EVAL,
                    args={"policy_uri": f"wandb://run/{run.run_id}"},
                ))

        # Schedule new training runs
        if available_training_slots > 0 and len(runs) < self.max_trials:
            jobs.append(JobDefinition(
                run_id=f"exp_{len(runs)+1:03d}",
                cmd="experiments.recipes.arena.train",
                type=JobTypes.LAUNCH_TRAINING,
                args={"run": f"exp_{len(runs)+1:03d}", "group": "my_experiment"},
                overrides={"trainer.total_timesteps": 1000000},
                gpus=4,
            ))

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        """Check if experiment is finished."""
        return len(runs) >= self.max_trials and all(
            r.status == JobStatus.COMPLETED for r in runs
        )
```

#### Understanding JobDefinition

The `JobDefinition` dataclass is the key to dispatching work:

```python
@dataclass
class JobDefinition:
    run_id: str                           # Unique identifier for this run
    cmd: str                               # Recipe entrypoint (e.g., "experiments.recipes.arena.train")
    gpus: int = 1                          # Number of GPUs required
    nodes: int = 1                         # Number of nodes required
    args: dict[str, Any] = {}              # Function arguments (e.g., run, group, policy_uri)
    overrides: dict[str, Any] = {}        # Config overrides (e.g., trainer.total_timesteps)
    type: JobTypes = JobTypes.LAUNCH_TRAINING  # Either LAUNCH_TRAINING or LAUNCH_EVAL
    metadata: dict[str, Any] = {}         # Additional metadata for tracking
```

The `JobTypes` enum has two values:
- `JobTypes.LAUNCH_TRAINING` - Training jobs that count against resource limits
- `JobTypes.LAUNCH_EVAL` - Evaluation jobs that can run without resource constraints

For more complex job patterns, see the models in `metta/adaptive/models.py`.

#### Step 2: Create a Scheduler Configuration (Optional)

Create a configuration class for your scheduler if it needs parameters:

```python
from metta.mettagrid.config import Config
from typing import Any

class MySchedulerConfig(Config):
    """Configuration for my custom scheduler."""
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    max_trials: int = 10
    gpus: int = 4
    experiment_id: str = "my_experiment"
    train_overrides: dict[str, Any] = {}
```

#### Step 3: Use AdaptiveController Directly

Create and run your experiment using AdaptiveController:

```python
from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.stores import WandbStore
from metta.adaptive.dispatcher import SkypilotDispatcher

# Create your scheduler
config = MySchedulerConfig(max_trials=10)
scheduler = MyScheduler(config)

# Set up components
store = WandbStore(entity="my-entity", project="my-project")
dispatcher = SkypilotDispatcher()

# Create and run controller
controller = AdaptiveController(
    experiment_id="my_experiment",
    scheduler=scheduler,
    dispatcher=dispatcher,
    store=store,
    config=AdaptiveConfig(
        max_parallel=4,
        monitoring_interval=60,
        resume=False
    )
)

# Run the experiment
controller.run()
```

## Running an Adaptive Experiment

### Using a Specialized Tool

For tools like `SweepTool`, use the standard tools runner:

```bash
# Bayesian hyperparameter optimization
uv run ./tools/run.py experiments.recipes.arena.sweep \
    experiment_id=my_sweep \
    max_trials=20 \
    batch_size=4
```

### Using AdaptiveController Directly

For custom experiments, create a Python script:

```python
# my_experiment.py
from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.stores import WandbStore
from metta.adaptive.dispatcher import SkypilotDispatcher
from my_schedulers import MyCustomScheduler

if __name__ == "__main__":
    scheduler = MyCustomScheduler(max_trials=10)
    store = WandbStore(entity="my-entity", project="my-project")
    dispatcher = SkypilotDispatcher()

    controller = AdaptiveController(
        experiment_id="my_custom_exp",
        scheduler=scheduler,
        dispatcher=dispatcher,
        store=store,
        config=AdaptiveConfig(max_parallel=4)
    )
    controller.run()
```

Then run it:
```bash
python my_experiment.py
```

## Advanced Data Tracking Between Runs

### Using Hooks

The `AdaptiveController` provides a hook for advanced data tracking:

**`on_eval_completed`**: Called after a run's evaluation completes
```python
def on_eval_completed(run: RunInfo, store: Store, all_runs: list[RunInfo]) -> None:
    """Process evaluation results and update run summaries.

    Common use case: Extract metrics and format them for the optimizer.
    For example, SweepTool uses this to extract scores and update observations
    in the sweep/ namespace.
    """
```

Example usage with AdaptiveController:
```python
def my_on_eval_completed(run, store, all_runs):
    # Extract metrics from evaluation
    summary = run.summary or {}
    score = summary.get("evaluator/eval_arena/score")

    # Update run summary with formatted observations
    store.update_run_summary(run.run_id, {
        "sweep/score": score,
        "sweep/cost": run.cost or 0,
    })

controller = AdaptiveController(
    experiment_id="my_exp",
    scheduler=scheduler,
    dispatcher=dispatcher,
    store=store,
    config=config,
    on_eval_completed=my_on_eval_completed
)

### Using Experiment State

There is an `ExperimentState` protocol in the codebase, although its role/scope has not been fully determined. We will implement state persistence in a future iteration to save state knowledge in case of spurious failures.

## Observation Format for Optimization

When using the adaptive framework for optimization (e.g., Bayesian optimization with SweepTool), observations follow a specific format:

### The sweep/ Namespace Convention

Optimization-related data is stored in the `sweep/` namespace in run summaries:

- `sweep/score`: The metric value being optimized
- `sweep/cost`: The computational cost of the run (e.g., time, FLOPs)
- `sweep/suggestion`: The hyperparameters used for this run

Example of properly formatted observations:

```python
# In your on_eval_completed hook
def on_eval_completed(run, store, all_runs):
    summary = run.summary or {}

    # Extract the metric you're optimizing
    score = summary.get("evaluator/eval_arena/score")

    # Store in sweep namespace for optimizer
    store.update_run_summary(run.run_id, {
        "sweep/score": score,
        "sweep/cost": run.cost or 0,
        "sweep/suggestion": run.metadata.get("sweep/suggestion", {})
    })
```

### Passing Suggestions to Runs

Suggestions from the optimizer should be passed via job metadata to ensure persistence:

```python
def schedule(self, runs, available_training_slots):
    # Generate suggestions from optimizer
    suggestions = self.optimizer.suggest(observations, n_suggestions=batch_size)

    jobs = []
    for suggestion in suggestions:
        job = create_training_job(
            run_id=run_id,
            train_overrides=suggestion,  # Apply as config overrides
            ...
        )
        # Store in metadata for persistence via initial_summary
        job.metadata["sweep/suggestion"] = suggestion
        jobs.append(job)

    return jobs
```

The AdaptiveController automatically passes job.metadata as initial_summary when initializing training runs, ensuring suggestions are persisted even with WandB's eventual consistency.

## Controller Statelessness and Safety

### Current Limitations

So the controller is stateless - how can you avoid double dispatch and other issues?

Currently there are very thin guards against duplicate errors, but they are not rock solid. This should be improved on, and we will probably need to tweak our design to be able to run auto-adaptive experiments with peace of mind.

### Recommended Approach

In the meantime, we recommend implementing your own safety layer through the scheduler state. To do this, simply subclass `ExperimentState` and make it known to your scheduler:

```python
from metta.adaptive.protocols import ExperimentState

class MySchedulerState(ExperimentState):
    dispatched_runs: set[str] = set()
    completed_evaluations: set[str] = set()

    def model_dump(self) -> dict[str, Any]:
        return {
            "dispatched_runs": list(self.dispatched_runs),
            "completed_evaluations": list(self.completed_evaluations),
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "MySchedulerState":
        state = cls()
        state.dispatched_runs = set(data.get("dispatched_runs", []))
        state.completed_evaluations = set(data.get("completed_evaluations", []))
        return state
```

## Design Philosophy and Future Directions

### The Tooling Philosophy

The adaptive framework takes a hybrid approach to tooling:

1. **Specialized tools for common patterns**: Tools like `SweepTool` provide opinionated, easy-to-use interfaces for specific experiment types. These tools are fully serializable and integrate with the tools runner.

2. **Direct controller usage for flexibility**: The `AdaptiveController` can be used directly in Python scripts for maximum flexibility and custom experiments.

This design recognizes that:
- Some experiments naturally fit into reusable patterns (sweeps, NAS, validation runs)
- Other experiments are highly custom and benefit from direct control
- What matters most is that the jobs launched by experiments are serializable, which is always the case

The framework prioritizes composability and flexibility over forcing everything into a tool pattern.

## Architecture Overview

The adaptive experiments framework consists of several key components:

### Core Components

```
Two Entry Points:
┌──────────────────┐         ┌──────────────────┐
│ Specialized Tool │         │   Direct Usage   │
│  (e.g. SweepTool)│         │  (Python Script) │
└────────┬─────────┘         └────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │                  │
         │ AdaptiveController
         │                  │
         └────────┬─────────┘
                  │
       ┌──────────┼──────────┬──────────┐
       ▼          ▼          ▼          ▼
  ┌─────────┐ ┌─────────┐ ┌──────┐ ┌──────────┐
  │Scheduler│ │  Store  │ │Config│ │Dispatcher│
  └─────────┘ │ (WandB) │ └──────┘ │  (Local/ │
              └─────────┘           │ Skypilot)│
                                    └──────────┘
```

**Entry Points**:
- **Specialized Tools** (e.g., `SweepTool`): Purpose-built tools for specific experiment types
- **Direct Usage**: Custom Python scripts using AdaptiveController directly

**AdaptiveController**: Main orchestration loop that fetches runs, calls scheduler, and dispatches jobs
**Scheduler**: Contains experiment logic and decides what jobs to run next
**Store**: Persistent storage for run state (currently WandB)
**Dispatcher**: Handles job execution (Local or Skypilot)
**Config**: AdaptiveConfig with settings like max_parallel, monitoring_interval

### Controller Lifecycle

The `AdaptiveController.run()` method implements the main experiment loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AdaptiveController.run()                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Initial Sleep       │ ◄── (if not resuming)
                    │ (monitoring_interval) │
                    └──────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                         MAIN LOOP START                          │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │    Fetch All Runs    │
                    │   from Store (WandB) │ ◄── Retry on failure
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Display Monitoring  │
                    │       Table          │ ◄── Shows run statuses
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Process Completed  │
                    │     Evaluations      │ ◄── Calls on_eval_completed
                    └──────────────────────┘         hook if provided
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Check Experiment    │
                    │     Complete?        │ ◄── scheduler.is_experiment_complete()
                    └──────────┬───────────┘
                         No │   │ Yes
                            │   └────────► EXIT
                            ▼
                    ┌──────────────────────┐
                    │ Calculate Available  │
                    │   Training Slots     │ ◄── max_parallel - active_runs
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Call Scheduler     │
                    │   schedule(runs,     │ ◄── Returns JobDefinition list
                    │   available_slots)   │
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Validate Resource   │
                    │    Constraints       │ ◄── Training jobs ≤ slots
                    └──────────────────────┘      Eval jobs unlimited
                                │
                                ▼
                    ┌──────────────────────────────────┐
                    │       Dispatch Jobs              │
                    ├──────────────────────────────────┤
                    │ For each JobDefinition:          │
                    │                                  │
                    │ if LAUNCH_TRAINING:              │
                    │   • dispatcher.dispatch(job)     │
                    │   • store.init_run() with        │
                    │     initial_summary=metadata     │
                    │                                  │
                    │ if LAUNCH_EVAL:                  │
                    │   • dispatcher.dispatch(job)     │
                    │   • store.update_run_summary()   │
                    └──────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │       Sleep          │
                    │ (monitoring_interval)│
                    └──────────────────────┘
                                │
                                └─────────► LOOP
```

#### Key Flow Details:

1. **Initial Sleep** (if not resuming)
   - Waits for `monitoring_interval` seconds
   - Prevents timeout on first WandB fetch

2. **Fetch Runs**
   - Retrieves all runs for the experiment from the Store
   - Includes retry logic for resilience

3. **Display Monitoring Table**
   - Shows current status of all runs
   - Updates every monitoring interval

4. **Process Evaluation Completions**
   - Calls `on_eval_completed` hook for newly evaluated runs
   - Sets `adaptive/post_eval_processed` flag to ensure idempotency
   - Records processing timestamp

5. **Check Experiment Completion**
   - Calls `scheduler.is_experiment_complete(runs)`
   - Exits loop if true

6. **Calculate Available Resources**
   - Counts runs with status `PENDING` or `IN_TRAINING`
   - Computes `available_training_slots` based on `config.max_parallel`

7. **Schedule New Jobs**
   - Calls `scheduler.schedule(runs, available_training_slots)`
   - Returns list of `JobDefinition` objects

8. **Validate Resource Constraints**
   - Ensures training jobs don't exceed available slots
   - Evaluation jobs have no resource limits

9. **Dispatch Jobs**
   - Sends jobs to dispatcher (Local or Skypilot)
   - Tracks dispatched jobs to prevent duplicates
   - Updates Store with run initialization or eval start

10. **Loop or Exit**
    - Continues until experiment complete
    - Handles keyboard interrupts gracefully

### Anatomy of a Scheduler

Schedulers are the most important part of the adaptive experiments framework. They encapsulate all experiment-specific logic.

#### Basic Example: Train and Eval

```python
class TrainAndEvalScheduler:
    """Simple scheduler: train jobs followed by eval jobs."""

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs = []

        # Priority 1: Eval completed training
        for run in runs:
            if run.status == JobStatus.TRAINING_DONE_NO_EVAL:
                jobs.append(create_eval_job(run_id=run.run_id, ...))

        # Priority 2: New training runs
        current_trials = len(runs)
        while available_training_slots > 0 and current_trials < self.max_trials:
            jobs.append(create_training_job(run_id=f"trial_{current_trials+1}", ...))
            available_training_slots -= 1
            current_trials += 1

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        return len(runs) >= self.max_trials and all(
            r.status == JobStatus.COMPLETED for r in runs
        )
```

#### Medium Example: Batched Synchronized Scheduler

```python
from metta.adaptive.models import JobStatus
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.adaptive.utils import create_training_job, create_eval_job

class BatchedSyncedOptimizingScheduler:
    """Scheduler that generates batches of suggestions synchronously."""

    def __init__(self, config: BatchedSyncedSchedulerConfig, state=None):
        self.config = config
        self.optimizer = ProteinOptimizer(config.protein_config)
        # Track state to prevent duplicate dispatches
        self.state = state or SchedulerState()

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs = []

        # Schedule evals first
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
        for run in eval_candidates:
            if run.run_id not in self.state.runs_in_eval:
                jobs.append(create_eval_job(run.run_id, ...))
                self.state.runs_in_training.discard(run.run_id)
                self.state.runs_in_eval.add(run.run_id)

        if jobs:
            return jobs  # Return early if evals needed

        # Barrier: wait until all runs complete before next batch
        if self.state.runs_in_training or self.state.runs_in_eval:
            return []  # Wait for all to finish

        # Get observations from completed runs (using sweep/ namespace)
        observations = []
        for run in runs:
            if run.status == JobStatus.COMPLETED and run.summary:
                obs = {
                    "score": run.summary.get("sweep/score", 0),
                    "cost": run.summary.get("sweep/cost", 0),
                    "suggestion": run.summary.get("sweep/suggestion", {})
                }
                observations.append(obs)

        # Generate new batch of suggestions
        suggestions = self.optimizer.suggest(observations, n_suggestions=self.config.batch_size)

        # Create training jobs with suggestions in metadata
        for i, suggestion in enumerate(suggestions[:available_training_slots]):
            run_id = f"batch_{len(runs)//self.config.batch_size}_{i}"
            job = create_training_job(
                run_id=run_id,
                train_overrides={**self.config.train_overrides, **suggestion},
                ...
            )
            # Store suggestion in metadata for persistence
            job.metadata["sweep/suggestion"] = suggestion
            jobs.append(job)
            self.state.runs_in_training.add(run_id)

        return jobs
```

#### Complex Example: Neural Architecture Search (Hypothetical)

```python
class NASScheduler:
    """Neural Architecture Search with progressive complexity."""

    def __init__(self, config: NASConfig):
        self.config = config
        self.architecture_space = self._build_search_space()
        self.performance_history = {}
        self.current_complexity_level = 1

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs = []

        # Analyze performance trends
        for run in runs:
            if run.status == JobStatus.COMPLETED:
                arch = self._extract_architecture(run)
                score = run.summary.get("observation/score", 0)
                self.performance_history[arch] = score

        # Adaptively adjust search strategy
        if self._should_increase_complexity():
            self.current_complexity_level += 1
            self._expand_search_space()

        # Prune poor performing architectures
        promising_archs = self._select_promising_architectures()

        # Generate new architectures based on promising ones
        for _ in range(min(available_training_slots, self.config.parallel_trials)):
            if len(runs) >= self.config.max_trials:
                break

            # Mutate or crossover existing architectures
            new_arch = self._generate_architecture(promising_archs)

            # Create training job with architecture config
            job = JobDefinition(
                run_id=f"nas_{len(runs)+1:04d}",
                cmd="experiments.recipes.nas.train",
                type=JobTypes.LAUNCH_TRAINING,
                overrides={
                    "agent.hidden_dims": new_arch["hidden_dims"],
                    "agent.num_layers": new_arch["num_layers"],
                    "agent.activation": new_arch["activation"],
                    "trainer.total_timesteps": self._get_timesteps_for_level(),
                },
                gpus=self._get_gpus_for_architecture(new_arch),
                metadata={"architecture": new_arch, "complexity_level": self.current_complexity_level}
            )
            jobs.append(job)

        return jobs

    def _should_increase_complexity(self) -> bool:
        """Decide if we should explore more complex architectures."""
        recent_improvements = self._calculate_recent_improvements()
        return recent_improvements < self.config.improvement_threshold

    def _generate_architecture(self, promising_archs: list) -> dict:
        """Generate new architecture via mutation or crossover."""
        # Complex logic for architecture generation
        ...
```

### Scheduler Responsibilities

A scheduler is responsible for:

1. **Job Scheduling**: Deciding what jobs to run next based on experiment state
2. **Resource Management**: Respecting `available_training_slots` constraint
3. **State Tracking**: Maintaining any experiment-specific state (though stateless is preferred)
4. **Completion Detection**: Determining when the experiment is finished
5. **Experiment Logic**: Implementing the core experimental strategy (optimization, validation, etc.)

## Creating Your Own Specialized Tool

If you have a reusable experiment pattern, consider creating a specialized tool like `SweepTool`:

```python
from metta.common.tool import Tool
from metta.adaptive import AdaptiveController, AdaptiveConfig
from metta.adaptive.stores import WandbStore
from metta.adaptive.dispatcher import SkypilotDispatcher

class MySpecializedTool(Tool):
    """Tool for my specific experiment type."""

    # Tool-specific configuration
    max_trials: int = 10
    experiment_id: str = "my_experiment"
    # ... other parameters

    def invoke(self, args: dict[str, str]) -> int | None:
        """Execute the experiment."""

        # Create scheduler with tool-specific logic
        scheduler = MySpecializedScheduler(
            max_trials=self.max_trials,
            # ... pass other config
        )

        # Set up components
        store = WandbStore(
            entity=self.wandb.entity,
            project=self.wandb.project
        )
        dispatcher = SkypilotDispatcher()

        # Optional: Define hooks
        def on_eval_completed(run, store, all_runs):
            # Tool-specific evaluation processing
            pass

        # Create and run controller
        controller = AdaptiveController(
            experiment_id=self.experiment_id,
            scheduler=scheduler,
            dispatcher=dispatcher,
            store=store,
            config=AdaptiveConfig(
                max_parallel=4,
                monitoring_interval=60
            ),
            on_eval_completed=on_eval_completed
        )

        # Run the experiment
        controller.run()
        return 0
```

### When to Create a Specialized Tool

Create a specialized tool when:
- You have a reusable experiment pattern (e.g., hyperparameter sweeps, architecture search)
- You want to provide a simplified interface for common use cases
- You need tool-specific configuration and validation
- You want to integrate with the tools runner (`./tools/run.py`)

Use AdaptiveController directly when:
- You're prototyping a new experiment type
- The experiment is a one-off or highly custom
- You need maximum flexibility in configuration
- You prefer Python scripts over the tools runner

## TODOs and Known Issues

### Planned Improvements

1. **State Persistence**: Implement robust state persistence to handle spurious failures
   - Save scheduler state periodically
   - Resume from last known state on restart

2. **Double Dispatch Prevention**: Strengthen guards against duplicate job dispatch
   - Implement transaction-like semantics
   - Add better idempotency checks

3. **Early Termination Support**: Add support for killing underperforming runs
   - Monitor run metrics in real-time
   - Cancel jobs based on performance criteria

4. **Hook System Refinement**: Stabilize and expand the hook system
   - Add more hook points in the lifecycle
   - Standardize hook signatures

5. **Tooling Philosophy**: Re-evaluate whether tooling is the right abstraction
   - Consider alternative architectures
   - Maintain serialization requirements

### Known Issues

1. **WandB Eventual Consistency**: WandB has eventual consistency which could cause missing data. This is mitigated by passing suggestions via initial_summary at run creation.
2. **WandB Timeouts**: First fetch from WandB may timeout on fresh experiments. The controller waits for monitoring_interval on first iteration to avoid this.
3. **Resource Counting**: Resource constraints only apply to training jobs, not evaluation jobs.
4. **Hook Failures**: Hook failures are logged but don't stop the experiment.
5. **State Persistence**: Scheduler state is not persisted across controller restarts. Some schedulers (like BatchedSyncedOptimizingScheduler) maintain their own state tracking to mitigate this.

### Recent Improvements

1. **Stateful Schedulers**: BatchedSyncedOptimizingScheduler now maintains state (runs_in_training, runs_in_eval, runs_completed) to prevent duplicate dispatches.
2. **Suggestion Persistence**: Suggestions are now passed via job.metadata and persisted as initial_summary to handle WandB eventual consistency.
3. **Completion Detection**: Fixed to properly count FAILED and STALE runs toward experiment limits.

## Contributing

When contributing to the adaptive experiments framework:

1. Follow the protocol-based design pattern
2. Ensure all schedulers are stateless when possible
3. Write comprehensive tests for new schedulers
4. Document any new hooks or state requirements
5. Keep jobs fully serializable

For questions or discussions about the architecture, please reach out to the team.