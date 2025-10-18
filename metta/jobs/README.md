# Job System Architecture

This document clarifies the roles and relationships between the key classes in the job infrastructure.

## Core Classes and Their Roles

### 1. Tool (from `metta/common/tool.py`)
**Role:** User-facing interface for defining work to be done

**What it is:**
- A Pydantic model describing a command to run (train, eval, analyze, etc.)
- Created by "tool maker" functions in recipe modules
- Contains high-level configuration (e.g., trainer config, sim config)
- Has an `invoke()` method that does the actual work

**Key characteristics:**
- Created at runtime from user CLI arguments
- Lives in memory during execution
- NOT serialized for remote execution
- Examples: `TrainTool`, `EvalTool`, `AnalyzeTool`, `ExperimentTool`

**Usage:**
```python
# Tool maker function (in recipe module)
def train(run: str, system: SystemConfig = ...) -> TrainTool:
    return TrainTool(
        run=run,
        system=system,
        trainer=TrainerConfig(...),
        sim=SimConfig(...),
    )

# Tool runner invokes it
tool = train(run="my_run")
tool.invoke({})  # Does the training work
```

### 2. JobSpec (from `metta/jobs/models.py`)
**Role:** Serialization format for launching jobs (local or remote)

**What it is:**
- A lightweight dataclass describing HOW to launch a Tool
- Contains: module path + args + overrides
- Serializable to JSON
- Execution-agnostic (can run locally or remotely)

**Key characteristics:**
- Can be serialized/deserialized
- Contains NO business logic, just data
- Converts to RemoteJob or LocalJob at launch time
- Used by: experiments, release system, adaptive sweeps

**Usage:**
```python
# Define what to launch
spec = JobSpec(
    name="my_training_job",
    module="experiments.recipes.arena.train",  # Path to tool maker
    args={"run": "my_run"},                     # Args to tool maker
    overrides={"trainer.total_timesteps": 1e6}, # Config overrides
    execution="remote",
    gpus=4,
)

# Later: convert to RemoteJob for execution
remote_job = RemoteJob(**spec.to_remote_job_args(log_dir="logs/"))
```

**Relationship to Tool:**
```
JobSpec describes --> Tool maker function --> Creates Tool instance --> Runs invoke()
                     (module path)           (at launch time)        (does work)
```

### 3. Job (from `metta/jobs/runner.py`)
**Role:** Execution wrapper for running jobs and capturing results

**What it is:**
- Abstract base class with concrete implementations:
  - `RemoteJob`: Executes via SkyPilot in the cloud
  - `LocalJob`: Executes via subprocess on local machine
- Handles: submission, status polling, log capture, cancellation
- Returns `JobResult` when complete

**Key characteristics:**
- Created from JobSpec (via `to_remote_job_args()` or `to_local_job_args()`)
- Lives during job execution
- NOT serialized
- Provides uniform interface regardless of execution mode

**Usage:**
```python
# Remote execution
job = RemoteJob(
    name="training",
    module="experiments.recipes.arena.train",
    args=["run=my_run", "trainer.total_timesteps=1000000"],
    base_args=["--gpus=4"],
)
job.submit()
result = job.wait(stream_output=True)

# Local execution
job = LocalJob(
    name="training",
    cmd=["uv", "run", "./tools/run.py", "experiments.recipes.arena.train", "run=my_run"],
)
result = job.wait(stream_output=True)
```

**Relationship to JobSpec:**
```
JobSpec.to_remote_job_args() --> RemoteJob instance --> submit() --> wait() --> JobResult
JobSpec.to_local_job_args()  --> LocalJob instance  --> submit() --> wait() --> JobResult
```

### 4. ExperimentTool (from `metta/experiment/tool.py`)
**Role:** Special Tool for managing groups of jobs

**What it is:**
- A Tool subclass (inherits from `metta.common.tool.Tool`)
- Contains a list of JobSpecs
- Has multiple modes: launch, attach, cancel, monitor, report
- Orchestrates launching/monitoring multiple jobs

**Key characteristics:**
- IS a Tool (user defines it as tool maker function)
- CONTAINS JobSpecs (list of jobs to launch)
- Delegates execution to ExperimentLauncher
- Different from other Tools: doesn't do direct work, orchestrates jobs

**Usage:**
```python
# Define experiment (tool maker function)
def my_experiment() -> ExperimentTool:
    return ExperimentTool(
        name="lr_sweep",
        jobs=[
            JobSpec(name="lr_001", module="...", args={"lr": 0.001}),
            JobSpec(name="lr_003", module="...", args={"lr": 0.003}),
        ],
    )

# User runs: ./tools/run.py experiments.user.my_module.my_experiment
# Tool runner creates ExperimentTool instance and calls invoke()
# invoke() creates ExperimentLauncher which creates RemoteJob/LocalJob instances
```

**Relationship to others:**
```
ExperimentTool (is a) Tool
              (contains) List[JobSpec]
              (delegates to) ExperimentLauncher
                            (creates) RemoteJob or LocalJob instances
```

## Class Hierarchy Summary

```
Tool (abstract base)
├── TrainTool          - Runs training directly
├── EvalTool           - Runs evaluation directly
├── AnalyzeTool        - Runs analysis directly
└── ExperimentTool     - Orchestrates multiple jobs
    └── uses: List[JobSpec]  - Describes what jobs to launch
                └── converts to: RemoteJob or LocalJob

Job (abstract base)
├── RemoteJob          - Executes job via SkyPilot
└── LocalJob           - Executes job via subprocess
```

## Data Flow: From User Input to Job Execution

### Single Tool Execution
```
1. User CLI: ./tools/run.py train arena run=my_run
2. Tool Registry: finds tool maker "experiments.recipes.arena.train"
3. Tool Maker: creates TrainTool instance with config
4. Tool.invoke(): runs training work directly
```

### Experiment (Multi-Job) Execution
```
1. User CLI: ./tools/run.py experiments.user.my_exp.my_experiment
2. Tool Registry: finds tool maker "experiments.user.my_exp.my_experiment"
3. Tool Maker: creates ExperimentTool instance with List[JobSpec]
4. ExperimentTool.invoke(): delegates to ExperimentLauncher
5. ExperimentLauncher: for each JobSpec:
   a. Convert: JobSpec -> RemoteJob or LocalJob (based on execution field)
   b. Submit job (remote: SkyPilot, local: subprocess)
   c. Track job_id and status in ExperimentState
6. Jobs run asynchronously, state updated periodically
```

## When to Use Each Class

### Use Tool when:
- Defining user-facing commands (train, eval, analyze, etc.)
- Need Pydantic validation of configuration
- Work happens in the `invoke()` method

### Use JobSpec when:
- Need to serialize a job for later execution
- Want execution-agnostic job description
- Building systems that launch multiple jobs (experiments, sweeps)

### Use RemoteJob/LocalJob when:
- Actually executing a job
- Need to track job status and capture logs
- Want uniform interface for remote vs local execution

### Use ExperimentTool when:
- Orchestrating 2-20 related jobs
- Need to track multiple jobs as a group
- Want experiment-level lifecycle management (launch, monitor, cancel)

## Key Design Decisions

### Why not just use Tool directly for remote execution?
- Tools contain heavy configuration objects (trainer config, sim config)
- Tools have methods and validation logic that shouldn't be serialized
- We need a lightweight format (JobSpec) that's easy to persist and transmit

### Why have both JobSpec and RemoteJob?
- **JobSpec**: Serializable data - describes WHAT to launch
- **RemoteJob**: Runtime object - handles HOW to execute
- Separation allows JobSpec to be saved/loaded while RemoteJob handles execution

### Why is ExperimentTool a Tool instead of just using JobSpec?
- Experiments ARE tools from user's perspective (they invoke them via CLI)
- Inheriting from Tool gives us:
  - Automatic CLI integration
  - Pydantic validation
  - Consistent UX with other tools
- ExperimentTool contains JobSpecs, doesn't replace them

### Future: Could we unify more?
Potential areas to simplify in the future:
- **JobSpec could generate Tools**: Add `JobSpec.to_tool()` that reconstructs Tool instance
- **Tools could generate JobSpecs**: Add `Tool.to_job_spec()` for serialization
- This would make the relationship more bidirectional and explicit

## Examples

### Example 1: Direct Tool Execution
```python
# User runs: uv run ./tools/run.py train arena run=my_run

# Tool maker creates Tool
tool = TrainTool(run="my_run", trainer=..., sim=...)

# Tool executes directly (no JobSpec, no RemoteJob)
tool.invoke({})  # Does training work
```

### Example 2: Remote Job via JobSpec
```python
# System wants to run training remotely
spec = JobSpec(
    name="train_lr001",
    module="experiments.recipes.arena.train",
    args={"run": "my_run"},
    execution="remote",
)

# Convert to RemoteJob and execute
job = RemoteJob(**spec.to_remote_job_args("logs/"))
job.submit()
result = job.wait()
```

### Example 3: Experiment with Multiple Jobs
```python
# User defines experiment
def my_experiment() -> ExperimentTool:
    return ExperimentTool(
        name="lr_sweep",
        jobs=[
            JobSpec(name="lr_001", module="...", execution="remote"),
            JobSpec(name="lr_003", module="...", execution="remote"),
        ],
    )

# User runs: uv run ./tools/run.py experiments.user.my_module.my_experiment
# Tool runner creates ExperimentTool instance
# ExperimentTool.invoke() creates ExperimentLauncher
# Launcher creates RemoteJob for each JobSpec
# Each RemoteJob submits to SkyPilot and tracks status
```
