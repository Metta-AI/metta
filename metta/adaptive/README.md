# Adaptive Experiments Framework

## What is the Adaptive Experiments Framework?

The adaptive experiments framework is a series of modules and protocols aiming to simplify the development, scaling, and execution of long-running experiments. It provides a clean, protocol-based architecture for orchestrating complex experimental workflows where decisions are made dynamically based on historical data and experiment state.

## When Should I Use Adaptive Experiments?

Adaptive experiments shine when your experiment needs:

1. **Many runs and/or decisions throughout an experiment** - Managing dozens or hundreds of training runs with complex dependencies
2. **Historical data in-experiment to inform decisions** - Using results from completed runs to guide future runs

The canonical adaptive experiment is, of course, **hyperparameter sweeps**, which is where this architecture originated from. However, the logic is flexible and lends itself to a vast array of experiment types:

- **Validating runs**: You may choose to write an experiment which picks out the best N runs of a group of experiments, and re-runs the training over different seeds for validation. Additionally, you may wish to cancel remaining seed runs if you notice that performance across the first K seeds is actually terrible.
- **Neural Architecture Search**: This becomes possible to write and test over 100s of runs as an adaptive experiment. Simply write a scheduler which produces different trainer configs.
- **Experiment batch with early termination** *(Not yet supported)*: You may launch a bunch of runs simultaneously, monitor their performance, and decide to kill aforementioned jobs if their performance is sub-par.

## How Do I Write an Adaptive Experiment?

Writing an adaptive experiment involves several steps:

### 1. Write a Scheduler

The scheduler is the brain of your experiment. It implements the `ExperimentScheduler` protocol with two required methods:

```python
from metta.adaptive.protocols import ExperimentScheduler
from metta.adaptive.models import JobDefinition, JobTypes, RunInfo

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

For more complex job patterns, see the models in `metta/sweep/models.py`.

### 2. Create a Scheduler Configuration

Create a Pydantic configuration class for your scheduler:

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
    # Add any other parameters your scheduler needs
```

### 3. Add Your Scheduler to the Tool

Update `metta/tools/adaptive.py`:

1. Add an enum value to `SchedulerType`:
```python
class SchedulerType(StrEnum):
    TRAIN_AND_EVAL = "train_and_eval"
    BATCHED_SYNCED = "batched_synced"
    MY_SCHEDULER = "my_scheduler"  # Add your scheduler type
```

2. Add a case in `AdaptiveTool._create_scheduler()`:
```python
def _create_scheduler(self, scheduler_type: SchedulerType, config_dict: dict) -> ExperimentScheduler:
    if scheduler_type == SchedulerType.MY_SCHEDULER:
        from metta.adaptive.schedulers.my_scheduler import MyScheduler, MySchedulerConfig
        config = MySchedulerConfig.model_validate(config_dict)
        return MyScheduler(config)
    # ... other cases
```

### 4. Create an Experiment Entrypoint

Create a recipe function that returns an `AdaptiveTool`:

```python
# experiments/recipes/adaptive/my_experiment.py
from metta.tools.adaptive import AdaptiveTool, SchedulerType, DispatcherType
from metta.adaptive.schedulers.my_scheduler import MySchedulerConfig

def my_experiment(
    run: str | None = None,  # Dispatcher passes this
    recipe_module: str = "experiments.recipes.arena",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    max_trials: int = 10,
    gpus: int = 4,
    experiment_id: str = "my_experiment",
    dispatcher_type: str = "skypilot",
    resume: bool = False,
) -> AdaptiveTool:
    """My custom adaptive experiment."""

    # Parse dispatcher type
    dispatcher_enum = DispatcherType.SKYPILOT if dispatcher_type == "skypilot" else DispatcherType.LOCAL

    # Create scheduler config
    scheduler_config = MySchedulerConfig(
        recipe_module=recipe_module,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        max_trials=max_trials,
        gpus=gpus,
        experiment_id=experiment_id,
    )

    # Create and return the tool
    return AdaptiveTool(
        scheduler_type=SchedulerType.MY_SCHEDULER,
        scheduler_config=scheduler_config.model_dump(),
        dispatcher_type=dispatcher_enum,
        config=AdaptiveConfig(
            max_parallel=4,
            monitoring_interval=60,
            resume=resume,
        ),
        experiment_id=experiment_id,
    )
```

## Running an Adaptive Experiment

Launch your experiment using the tools runner:

```bash
# Basic train & eval experiment
uv run ./tools/run.py experiments.recipes.adaptive.train_and_eval.train_and_eval \
    run=ak.sanity_adaptive_experiment \
    max_trials=4 \
    recipe_module=experiments.recipes.navigation \
    train_entrypoint=train \
    eval_entrypoint=evaluate \
    resume=False \
    experiment_id=ak.tenavpoc.91443
```

The entrypoint structure follows this pattern:
1. The recipe function is called with command-line arguments
2. It creates the appropriate scheduler configuration
3. It returns an `AdaptiveTool` configured with your scheduler
4. The tool's `run()` method is called automatically
5. The adaptive controller begins orchestrating your experiment

## Advanced Data Tracking Between Runs

### Using Hooks

The `AdaptiveController` provides two hooks for advanced data tracking:

1. **`on_eval_completed`**: Called after a run's evaluation completes
   ```python
   def on_eval_completed(run: RunInfo, store: Store, all_runs: list[RunInfo]) -> None:
       """Process evaluation results and update run summaries."""
   ```

2. **`on_job_dispatch`**: Called after dispatching a job
   ```python
   def on_job_dispatch(job: JobDefinition, store: Store) -> None:
       """Track dispatched jobs or update metadata."""
   ```

The final format/number of hooks is TBD and may change rapidly in the coming weeks.

### Using Experiment State

There is an `ExperimentState` protocol in the codebase, although its role/scope has not been fully determined. We will implement state persistence in a future iteration to save state knowledge in case of spurious failures.

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

### The Tooling Question

Having to create a different enum for each scheduler and writing all that boilerplate is off-putting. Is there an alternative?

We are trying to abide by the tooling philosophy: every tool should be fully serializable. We are already breaking this somewhat with our callbacks, and it may very well be that **tooling** is not the right way to do this.

Since you are here, let me go into more detail: it seems reasonable that some experiments could become their own tools (for example, sweeps!) but generally speaking, the adaptive tool could live one layer below/above that and be plugged **into** tools when we deem that necessary. What matters most is that the jobs that are launched by the adaptive experiments are themselves serializable, which is always going to be the case anyway.

## Architecture Overview

The adaptive experiments framework consists of several key components:

### Core Components

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│                 │     │              │     │              │
│  AdaptiveTool   │────►│  Controller  │────►│  Scheduler   │
│                 │     │              │     │              │
└────────┬────────┘     └──────┬───────┘     └──────────────┘
         │                     │                      ▲
         │                     ▼                      │
         │              ┌──────────────┐              │
         │              │              │              │
         └─────────────►│    Store     │◄─────────────┘
                        │   (WandB)    │
                        │              │
                        └──────────────┘
                               ▲
                               │
                        ┌──────┴───────┐
                        │              │
                        │  Dispatcher  │
                        │              │
                        └──────────────┘
```

**AdaptiveTool**: Entry point that wires together all components and manages configuration
**Controller**: Main orchestration loop that fetches runs, calls scheduler, and dispatches jobs
**Scheduler**: Contains experiment logic and decides what jobs to run next
**Store**: Persistent storage for run state (currently WandB)
**Dispatcher**: Handles job execution (Local or Skypilot)

### Controller Lifecycle

The `AdaptiveController.run()` method implements the main experiment loop:

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
class BatchedSyncedOptimizingScheduler:
    """Scheduler that generates batches of suggestions synchronously."""

    def __init__(self, config: BatchedSyncedSchedulerConfig):
        self.config = config
        self.optimizer = ProteinOptimizer(config.protein_config)

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs = []

        # Schedule evals first
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
        for run in eval_candidates:
            jobs.append(create_eval_job(run.run_id, ...))

        if jobs:
            return jobs  # Return early if evals needed

        # Barrier: wait until all runs complete before next batch
        incomplete = [r for r in runs if r.status not in (JobStatus.COMPLETED, JobStatus.FAILED)]
        if incomplete:
            return []  # Wait for all to finish

        # Get observations from completed runs
        observations = []
        for run in runs:
            if run.status == JobStatus.COMPLETED and run.summary:
                obs = Observation(
                    score=run.summary.get("observation/score", 0),
                    cost=run.summary.get("observation/cost", 0),
                    suggestion=run.summary.get("observation/suggestion", {})
                )
                observations.append(obs)

        # Generate new batch of suggestions
        suggestions = self.optimizer.suggest(observations, n_suggestions=self.config.batch_size)

        # Create training jobs with suggestions
        for i, suggestion in enumerate(suggestions[:available_training_slots]):
            job = create_training_job(
                run_id=f"batch_{len(runs)//self.config.batch_size}_{i}",
                train_overrides={**self.config.train_overrides, **suggestion},
                ...
            )
            jobs.append(job)

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

1. **Controller State**: The controller maintains minimal state (only dispatched jobs), which can lead to issues on restart
2. **WandB Timeouts**: First fetch from WandB may timeout on fresh experiments
3. **Resource Counting**: Resource constraints only apply to training jobs, not evaluation
4. **Hook Failures**: Hook failures are logged but don't stop the experiment

### Code TODOs

Several TODOs are peppered throughout the codebase:
- Clean up run lifecycle management in `models.py`
- Improve error handling in dispatcher implementations
- Add more robust retry logic for Store operations
- Implement proper transaction semantics for state updates

## Contributing

When contributing to the adaptive experiments framework:

1. Follow the protocol-based design pattern
2. Ensure all schedulers are stateless when possible
3. Write comprehensive tests for new schedulers
4. Document any new hooks or state requirements
5. Keep jobs fully serializable

For questions or discussions about the architecture, please reach out to the team.