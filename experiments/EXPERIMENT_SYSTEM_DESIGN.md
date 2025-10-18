# Experiment System Technical Design

**Author:** Jack & Claude
**Date:** 2025-10-13
**Status:** Implementation Ready

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Core Data Structures](#core-data-structures)
4. [Key APIs](#key-apis)
5. [CLI User Experience](#cli-user-experience)
6. [Implementation Guide](#implementation-guide)
7. [Code Reuse Strategy](#code-reuse-strategy)

---

## Overview

The experiment system allows users to define and manage groups of 2-20 related training/evaluation jobs. It differs from sweeps (adaptive scheduling) and release validation (sequential gates) by being **declarative** - all jobs are defined upfront as parameter variations.

**Core Principles:**
- **Module paths are the serialization format** - No intermediate YAML configs
- **Maximum code reuse** - Share infrastructure with release system and job_runner
- **Instance-based tracking** - Each experiment run gets unique ID
- **Recipe-oriented** - Define experiments as Python functions returning ExperimentTool

---

## File Structure

### New Files to Create

```
metta/jobs/                          # NEW: Shared job infrastructure
├── __init__.py
├── models.py                        # JobSpec, enhanced JobResult
├── runner.py                        # RemoteJob, LocalJob (moved from devops/)
├── state.py                         # StateStore protocol, JSONStateStore
├── monitor.py                       # Shared monitoring utilities
└── metrics.py                       # WandB artifact extraction

metta/experiment/                    # NEW: Experiment-specific logic
├── __init__.py
├── tool.py                          # ExperimentTool class
├── state.py                         # Experiment state management
└── report.py                        # Post-hoc report generation

metta/tools/
└── experiment.py                    # NEW: Tool registration

experiments/
├── state/                           # NEW: Runtime state (gitignored)
│   └── {instance_id}.json          # Per-instance state files
├── logs/                            # NEW: Job logs (gitignored)
│   └── {instance_id}/
│       ├── job1.{job_id}.log
│       └── job2.{job_id}.log
└── user/
    └── example_experiment.py        # NEW: Example implementation
```

### Files to Modify

```
devops/job_runner.py
  - Add back-compat imports from metta.jobs.runner
  - Keep existing API surface unchanged

.gitignore
  + experiments/state/
  + experiments/logs/
  + experiments/reports/
```

---

## Core Data Structures

### 1. JobSpec - Universal Job Specification

**Location:** `metta/jobs/models.py`

This is THE fundamental data structure. It replaces:
- Old PR's `TrainingJobConfig`
- `adaptive.JobDefinition`
- Arguments to `release.Task` constructors

```python
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class JobSpec:
    """Universal job specification for all Metta job systems.

    This is our serialization format - everything needed to launch a job
    via tools/run.py or RemoteJob.

    Serialization:
        module="experiments.recipes.arena.train"
        args={"run": "my_run"}
        overrides={"trainer.total_timesteps": 1000000}

    Becomes:
        ./tools/run.py experiments.recipes.arena.train run=my_run trainer.total_timesteps=1000000
    """

    # Job identification
    name: str

    # Tool maker specification
    module: str  # e.g., "experiments.recipes.arena.train"

    # Arguments to tool maker function
    args: dict[str, Any] = field(default_factory=dict)

    # Config overrides (dotted paths)
    overrides: dict[str, Any] = field(default_factory=dict)

    # Infrastructure settings
    gpus: int = 1
    nodes: int = 1
    spot: bool = True
    timeout_s: int = 7200

    # Job type (for specialized behavior)
    job_type: Literal["train", "eval", "task"] = "train"

    # Metadata (experiment_id, tags, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_remote_job_args(self, log_dir: str) -> dict[str, Any]:
        """Convert to RemoteJob constructor arguments."""
        # Flatten args and overrides into single list
        arg_list = []
        for k, v in self.args.items():
            arg_list.append(f"{k}={v}")
        for k, v in self.overrides.items():
            arg_list.append(f"{k}={v}")

        base_args = [f"--gpus={self.gpus}", f"--nodes={self.nodes}"]
        if not self.spot:
            base_args.insert(0, "--no-spot")

        return {
            "name": self.name,
            "module": self.module,
            "args": arg_list,
            "timeout_s": self.timeout_s,
            "log_dir": log_dir,
            "base_args": base_args,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "module": self.module,
            "args": self.args,
            "overrides": self.overrides,
            "gpus": self.gpus,
            "nodes": self.nodes,
            "spot": self.spot,
            "timeout_s": self.timeout_s,
            "job_type": self.job_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        """Deserialize from JSON-compatible dict."""
        return cls(**data)
```

### 2. ExperimentState - Persistent State

**Location:** `metta/experiment/state.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
from pathlib import Path
import json

ExperimentStatus = Literal["pending", "running", "completed", "partial", "failed", "cancelled"]

@dataclass
class JobState:
    """State of a single job within an experiment."""

    name: str
    spec: JobSpec  # Full job specification

    # Runtime state
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    job_id: Optional[str] = None  # Skypilot job ID
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    exit_code: Optional[int] = None
    logs_path: Optional[str] = None

    # Extracted artifacts
    wandb_url: Optional[str] = None
    wandb_run_id: Optional[str] = None
    checkpoint_uri: Optional[str] = None

    # Metrics (for acceptance criteria)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "logs_path": self.logs_path,
            "wandb_url": self.wandb_url,
            "wandb_run_id": self.wandb_run_id,
            "checkpoint_uri": self.checkpoint_uri,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        # Reconstruct JobSpec
        data["spec"] = JobSpec.from_dict(data["spec"])
        return cls(**data)


@dataclass
class ExperimentState:
    """Complete state of an experiment instance."""

    # Identity
    experiment_id: str  # e.g., "lr_comparison_20251013_1430"
    recipe: str  # e.g., "experiments.user.lr_comparison.my_experiment"

    # Timestamps
    created_at: str
    updated_at: str

    # Overall status
    status: ExperimentStatus

    # Job states
    jobs: dict[str, JobState] = field(default_factory=dict)  # job_name -> JobState

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "recipe": self.recipe,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "jobs": {name: job.to_dict() for name, job in self.jobs.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentState":
        data["jobs"] = {name: JobState.from_dict(job_data)
                       for name, job_data in data["jobs"].items()}
        return cls(**data)

    def save(self, base_dir: Path = Path("experiments/state")):
        """Save state to JSON file."""
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{self.experiment_id}.json"

        self.updated_at = datetime.utcnow().isoformat(timespec="seconds")

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, instance_id: str, base_dir: Path = Path("experiments/state")) -> Optional["ExperimentState"]:
        """Load state from JSON file."""
        path = base_dir / f"{instance_id}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_job_status(self, job_name: str, **updates):
        """Update a job's state and save."""
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found in experiment")

        job_state = self.jobs[job_name]
        for key, value in updates.items():
            setattr(job_state, key, value)

        # Update overall experiment status
        self._update_overall_status()

        self.save()

    def _update_overall_status(self):
        """Compute overall experiment status from job states."""
        if not self.jobs:
            self.status = "pending"
            return

        statuses = [job.status for job in self.jobs.values()]

        if all(s == "completed" for s in statuses):
            self.status = "completed"
        elif any(s == "failed" for s in statuses):
            self.status = "failed"
        elif any(s == "cancelled" for s in statuses):
            self.status = "cancelled"
        elif any(s == "running" for s in statuses):
            self.status = "running"
        elif all(s in ("completed", "pending") for s in statuses):
            self.status = "partial"
        else:
            self.status = "pending"
```

### 3. ExperimentTool - Main Tool Class

**Location:** `metta/experiment/tool.py`

```python
from typing import Optional, Literal
from pydantic import Field
from datetime import datetime

from metta.common.tool import Tool
from metta.rl.system_config import SystemConfig
from metta.jobs.models import JobSpec
from metta.experiment.state import ExperimentState, JobState

class ExperimentTool(Tool):
    """Tool for managing groups of related training/evaluation jobs.

    Usage modes:
    - launch (default): Start new experiment instance
    - attach: Monitor existing instance
    - cancel: Cancel all jobs in instance
    - monitor: Show live status
    - report: Generate post-hoc analysis notebook

    Example:
        def my_experiment() -> ExperimentTool:
            return ExperimentTool(
                name="lr_comparison",
                jobs=[
                    JobSpec(
                        name="lr_0001",
                        module="experiments.recipes.arena.train",
                        args={"run": "lr_comparison.lr_0001"},
                        overrides={"trainer.optimizer.learning_rate": 0.0001},
                    ),
                    JobSpec(
                        name="lr_0003",
                        module="experiments.recipes.arena.train",
                        args={"run": "lr_comparison.lr_0003"},
                        overrides={"trainer.optimizer.learning_rate": 0.0003},
                    ),
                ],
            )
    """

    # Required fields
    name: str
    jobs: list[JobSpec]

    # Mode control (mutually exclusive)
    mode: Literal["launch", "attach", "cancel", "monitor", "report"] = "launch"

    # Instance selection
    instance_id: Optional[str] = None  # If None, auto-generate (launch) or use latest (other modes)

    # Launch options
    dry_run: bool = False  # Show what would be launched
    sequential: bool = False  # Launch jobs one at a time (default: parallel)

    # Monitor options
    refresh_interval: int = 10  # Seconds between status updates

    # System config
    system: SystemConfig = Field(default_factory=SystemConfig)

    def invoke(self, args: dict[str, str]) -> int:
        """Main entry point from tool runner."""

        # Route to appropriate handler based on mode
        if self.mode == "launch":
            return self._launch()
        elif self.mode == "attach":
            return self._attach()
        elif self.mode == "cancel":
            return self._cancel()
        elif self.mode == "monitor":
            return self._monitor()
        elif self.mode == "report":
            return self._report()
        else:
            print(f"Unknown mode: {self.mode}")
            return 1

    def _launch(self) -> int:
        """Launch new experiment instance."""
        from metta.experiment.launcher import ExperimentLauncher

        # Generate instance ID if not provided
        if self.instance_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.instance_id = f"{self.name}_{timestamp}"

        launcher = ExperimentLauncher(
            instance_id=self.instance_id,
            recipe=self.__class__.__module__ + "." + self.__class__.__name__,
            jobs=self.jobs,
            dry_run=self.dry_run,
            sequential=self.sequential,
        )

        return launcher.launch()

    def _attach(self) -> int:
        """Attach to existing experiment instance."""
        from metta.experiment.monitor import ExperimentMonitor

        # Find instance
        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        monitor = ExperimentMonitor(instance_id)
        return monitor.attach()

    def _cancel(self) -> int:
        """Cancel all jobs in experiment instance."""
        from metta.experiment.manager import ExperimentManager

        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        manager = ExperimentManager(instance_id)
        return manager.cancel_all()

    def _monitor(self) -> int:
        """Show live monitoring of experiment."""
        from metta.experiment.monitor import ExperimentMonitor

        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        monitor = ExperimentMonitor(
            instance_id,
            refresh_interval=self.refresh_interval,
        )
        return monitor.run()

    def _report(self) -> int:
        """Generate post-hoc analysis report."""
        from metta.experiment.report import generate_report

        instance_id = self._resolve_instance_id()
        if instance_id is None:
            print(f"No experiment instance found for {self.name}")
            return 1

        return generate_report(instance_id)

    def _resolve_instance_id(self) -> Optional[str]:
        """Resolve instance ID - use provided or find most recent."""
        if self.instance_id:
            return self.instance_id

        # Find most recent instance for this experiment name
        from pathlib import Path
        import re

        state_dir = Path("experiments/state")
        if not state_dir.exists():
            return None

        # Pattern: {name}_{timestamp}.json
        pattern = re.compile(rf"^{re.escape(self.name)}_(\d{{8}}_\d{{6}})\.json$")

        matches = []
        for path in state_dir.glob("*.json"):
            match = pattern.match(path.name)
            if match:
                timestamp = match.group(1)
                matches.append((timestamp, path.stem))

        if not matches:
            return None

        # Return most recent
        matches.sort(reverse=True)
        return matches[0][1]
```

---

## Key APIs

### ExperimentLauncher - Job Launch Logic

**Location:** `metta/experiment/launcher.py`

```python
from typing import Optional
from pathlib import Path
from datetime import datetime

from metta.jobs.models import JobSpec
from metta.jobs.runner import RemoteJob
from metta.experiment.state import ExperimentState, JobState

class ExperimentLauncher:
    """Handles launching all jobs in an experiment."""

    def __init__(
        self,
        instance_id: str,
        recipe: str,
        jobs: list[JobSpec],
        dry_run: bool = False,
        sequential: bool = False,
    ):
        self.instance_id = instance_id
        self.recipe = recipe
        self.jobs = jobs
        self.dry_run = dry_run
        self.sequential = sequential

        self.log_dir = Path("experiments/logs") / instance_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def launch(self) -> int:
        """Launch all jobs."""

        # Create initial state
        state = ExperimentState(
            experiment_id=self.instance_id,
            recipe=self.recipe,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            updated_at=datetime.utcnow().isoformat(timespec="seconds"),
            status="pending",
            jobs={
                job.name: JobState(name=job.name, spec=job)
                for job in self.jobs
            },
        )

        if self.dry_run:
            print(f"\n[DRY RUN] Would launch experiment: {self.instance_id}")
            print(f"Recipe: {self.recipe}")
            print(f"\nJobs ({len(self.jobs)}):")
            for job in self.jobs:
                print(f"\n  {job.name}:")
                print(f"    Module: {job.module}")
                print(f"    Args: {job.args}")
                print(f"    Overrides: {job.overrides}")
                print(f"    Resources: {job.gpus} GPU(s), {job.nodes} node(s)")
            return 0

        # Save initial state
        state.save()

        print(f"\n{'='*80}")
        print(f"Launching Experiment: {self.instance_id}")
        print(f"{'='*80}")
        print(f"Recipe: {self.recipe}")
        print(f"Jobs: {len(self.jobs)}")
        print(f"Mode: {'Sequential' if self.sequential else 'Parallel'}")
        print(f"State: experiments/state/{self.instance_id}.json")
        print(f"Logs: experiments/logs/{self.instance_id}/")
        print(f"{'='*80}\n")

        # Inject experiment metadata into all jobs
        for job in self.jobs:
            job.metadata["experiment_id"] = self.instance_id
            # Set WandB group for all jobs
            if "group" not in job.args:
                job.args["group"] = self.instance_id

        # Launch jobs
        if self.sequential:
            success = self._launch_sequential(state)
        else:
            success = self._launch_parallel(state)

        if success:
            print(f"\n✅ Successfully launched {len(self.jobs)} job(s)")
            print(f"\nMonitor: ./tools/run.py {self.recipe} mode=monitor")
            return 0
        else:
            print(f"\n❌ Failed to launch some jobs")
            return 1

    def _launch_parallel(self, state: ExperimentState) -> bool:
        """Launch all jobs in parallel."""
        all_success = True

        for job_spec in self.jobs:
            print(f"\n{'─'*80}")
            print(f"Launching: {job_spec.name}")
            print(f"{'─'*80}")

            success = self._launch_single_job(job_spec, state)
            if not success:
                all_success = False

        return all_success

    def _launch_sequential(self, state: ExperimentState) -> bool:
        """Launch jobs one at a time, waiting for each to complete."""
        # For v1, we'll still launch in parallel but this hook exists
        # for future enhancement
        return self._launch_parallel(state)

    def _launch_single_job(self, job_spec: JobSpec, state: ExperimentState) -> bool:
        """Launch a single job via RemoteJob."""

        try:
            # Create RemoteJob
            job = RemoteJob(**job_spec.to_remote_job_args(str(self.log_dir)))

            # Submit (non-blocking)
            job.submit()

            # Extract job_id
            if job._job_id:
                job_id = str(job._job_id)

                # Update state
                state.update_job_status(
                    job_spec.name,
                    status="running",
                    job_id=job_id,
                    started_at=datetime.utcnow().isoformat(timespec="seconds"),
                    logs_path=str(job._get_log_path()),
                )

                print(f"✓ Launched successfully (Job ID: {job_id})")
                return True
            else:
                print(f"✗ Failed to get job ID")
                state.update_job_status(job_spec.name, status="failed")
                return False

        except Exception as e:
            print(f"✗ Launch failed: {e}")
            state.update_job_status(job_spec.name, status="failed")
            return False
```

### ExperimentMonitor - Live Status Display

**Location:** `metta/experiment/monitor.py`

```python
import time
from pathlib import Path
from typing import Optional

from metta.experiment.state import ExperimentState, JobState
from metta.jobs.monitor import format_job_table, poll_job_status
from metta.jobs.metrics import extract_wandb_info_from_logs

class ExperimentMonitor:
    """Monitor experiment status and display live updates."""

    def __init__(self, instance_id: str, refresh_interval: int = 10):
        self.instance_id = instance_id
        self.refresh_interval = refresh_interval
        self.state = ExperimentState.load(instance_id)

        if self.state is None:
            raise ValueError(f"Experiment instance not found: {instance_id}")

    def run(self) -> int:
        """Run live monitoring loop."""
        print(f"\nMonitoring experiment: {self.instance_id}")
        print(f"Press Ctrl+C to exit\n")

        try:
            while True:
                self._refresh_and_display()

                # Check if complete
                if self.state.status in ("completed", "failed", "cancelled"):
                    print(f"\nExperiment {self.state.status.upper()}")
                    return 0 if self.state.status == "completed" else 1

                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            return 0

    def attach(self) -> int:
        """Attach and stream logs from running jobs."""
        from metta.jobs.runner import RemoteJob

        print(f"\nAttaching to experiment: {self.instance_id}\n")

        # Find first running job and attach to it
        for job_name, job_state in self.state.jobs.items():
            if job_state.status == "running" and job_state.job_id:
                print(f"Attaching to job: {job_name} (Job ID: {job_state.job_id})")
                print(f"{'='*80}\n")

                # Create RemoteJob and attach
                job = RemoteJob(
                    name=job_name,
                    module=job_state.spec.module,
                    args=[f"{k}={v}" for k, v in job_state.spec.args.items()],
                    job_id=int(job_state.job_id),
                    log_dir=str(Path("experiments/logs") / self.instance_id),
                )

                # Stream logs
                result = job.wait(stream_output=True)

                # Update state with result
                self.state.update_job_status(
                    job_name,
                    status="completed" if result.exit_code == 0 else "failed",
                    exit_code=result.exit_code,
                    completed_at=datetime.utcnow().isoformat(timespec="seconds"),
                )

                return result.exit_code

        print("No running jobs to attach to")
        return 1

    def _refresh_and_display(self):
        """Refresh job states and display table."""
        import os

        # Reload state from disk (may have been updated by other process)
        self.state = ExperimentState.load(self.instance_id)

        # Poll Skypilot for job statuses
        for job_name, job_state in self.state.jobs.items():
            if job_state.status == "running" and job_state.job_id:
                # Check Skypilot status
                sky_status = poll_job_status(job_state.job_id)

                if sky_status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    # Job completed - update state
                    new_status = {
                        "SUCCEEDED": "completed",
                        "FAILED": "failed",
                        "CANCELLED": "cancelled",
                    }[sky_status]

                    self.state.update_job_status(
                        job_name,
                        status=new_status,
                        exit_code=0 if sky_status == "SUCCEEDED" else 1,
                        completed_at=datetime.utcnow().isoformat(timespec="seconds"),
                    )

                    # Try to extract WandB info from logs
                    if job_state.logs_path and Path(job_state.logs_path).exists():
                        wandb_info = extract_wandb_info_from_logs(job_state.logs_path)
                        if wandb_info:
                            self.state.update_job_status(
                                job_name,
                                wandb_url=wandb_info["url"],
                                wandb_run_id=wandb_info["run_id"],
                                checkpoint_uri=wandb_info["checkpoint_uri"],
                            )

        # Clear screen and display table
        os.system('clear' if os.name == 'posix' else 'cls')

        print(f"\n{'='*80}")
        print(f"Experiment: {self.instance_id}")
        print(f"Status: {self.state.status.upper()}")
        print(f"Updated: {self.state.updated_at}")
        print(f"{'='*80}\n")

        # Display job table
        self._display_job_table()

        print(f"\nRefreshing every {self.refresh_interval}s (Ctrl+C to stop)")

    def _display_job_table(self):
        """Display formatted table of job states."""
        from metta.common.util.text_styles import green, yellow, red, blue

        # Header
        print(f"{'Job Name':<30} {'Status':<15} {'Job ID':<10} {'WandB Run':<30}")
        print(f"{'─'*30} {'─'*15} {'─'*10} {'─'*30}")

        # Rows
        for job_name, job_state in self.state.jobs.items():
            # Status with color
            status_str = job_state.status.upper()
            if job_state.status == "completed":
                status_str = green(status_str)
            elif job_state.status == "failed":
                status_str = red(status_str)
            elif job_state.status == "running":
                status_str = yellow(status_str)
            else:
                status_str = blue(status_str)

            job_id_str = job_state.job_id or "-"
            wandb_str = job_state.wandb_run_id or "-"

            print(f"{job_name:<30} {status_str:<24} {job_id_str:<10} {wandb_str:<30}")

        print()
```

---

## CLI User Experience

### Simple Commands

#### 1. Launch New Experiment

```bash
# Launch with default settings
./tools/run.py experiments.user.my_experiment.my_experiment_maker

# Explicit launch mode
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=launch

# Dry run (see what would be launched)
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=launch dry_run=true
```

**What happens:**
- Generates instance ID: `my_experiment_20251013_1430`
- Creates state file: `experiments/state/my_experiment_20251013_1430.json`
- Launches all jobs to Skypilot in parallel
- Each job gets WandB group = instance_id
- Prints job IDs and monitoring command

**Output:**
```
================================================================================
Launching Experiment: my_experiment_20251013_1430
================================================================================
Recipe: experiments.user.my_experiment.my_experiment_maker
Jobs: 3
Mode: Parallel
State: experiments/state/my_experiment_20251013_1430.json
Logs: experiments/logs/my_experiment_20251013_1430/
================================================================================

────────────────────────────────────────────────────────────────────────────────
Launching: job1
────────────────────────────────────────────────────────────────────────────────
✓ Launched successfully (Job ID: 12345)

────────────────────────────────────────────────────────────────────────────────
Launching: job2
────────────────────────────────────────────────────────────────────────────────
✓ Launched successfully (Job ID: 12346)

────────────────────────────────────────────────────────────────────────────────
Launching: job3
────────────────────────────────────────────────────────────────────────────────
✓ Launched successfully (Job ID: 12347)

✅ Successfully launched 3 job(s)

Monitor: ./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=monitor
```

#### 2. Monitor Running Experiment

```bash
# Monitor latest instance of experiment
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=monitor

# Monitor specific instance
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=monitor instance_id=my_experiment_20251013_1430
```

**What happens:**
- Loads state from disk
- Polls Skypilot for job statuses
- Refreshes display every 10s
- Auto-exits when all jobs complete

**Output:**
```
================================================================================
Experiment: my_experiment_20251013_1430
Status: RUNNING
Updated: 2025-10-13T14:45:23
================================================================================

Job Name                       Status          Job ID     WandB Run
────────────────────────────── ─────────────── ────────── ──────────────────────────────
job1                           COMPLETED       12345      my_experiment.job1
job2                           RUNNING         12346      my_experiment.job2
job3                           PENDING         12347      -

Refreshing every 10s (Ctrl+C to stop)
```

#### 3. Attach to Running Job

```bash
# Attach to first running job and stream logs
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=attach
```

**What happens:**
- Finds first running job in experiment
- Attaches to it via RemoteJob
- Streams logs to console
- Updates state when job completes

#### 4. Cancel All Jobs

```bash
# Cancel all jobs in latest instance
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=cancel

# Cancel specific instance
./tools/run.py experiments.user.my_experiment.my_experiment_maker mode=cancel instance_id=my_experiment_20251013_1430
```

**What happens:**
- Loads state
- Calls `sky jobs cancel` for each running job
- Updates state to mark jobs as cancelled

### Complex Workflows

#### Workflow 1: Parameter Sweep with Evaluation

```python
# experiments/user/lr_sweep.py
from operator import ge
from metta.experiment.tool import ExperimentTool
from metta.jobs.models import JobSpec

def lr_sweep() -> ExperimentTool:
    """Train 3 learning rates and evaluate each."""

    jobs = []

    # Training jobs
    for lr in [0.0001, 0.0003, 0.001]:
        lr_str = f"lr_{int(lr*10000):04d}"

        # Training
        train_job = JobSpec(
            name=f"train_{lr_str}",
            module="experiments.recipes.arena.train",
            args={"run": f"lr_sweep.{lr_str}"},
            overrides={
                "trainer.optimizer.learning_rate": lr,
                "trainer.total_timesteps": 100_000_000,
            },
            gpus=4,
            nodes=1,
            metadata={"learning_rate": lr},
        )
        jobs.append(train_job)

        # Evaluation (manual dependency - evaluate after training completes)
        eval_job = JobSpec(
            name=f"eval_{lr_str}",
            module="experiments.recipes.arena.evaluate",
            args={
                # Note: This is a template - actual URI resolved from WandB
                "policy_uri": f"wandb://run/lr_sweep.{lr_str}",
            },
            overrides={},
            job_type="eval",
            metadata={"training_job": f"train_{lr_str}"},
        )
        jobs.append(eval_job)

    return ExperimentTool(
        name="lr_sweep",
        jobs=jobs,
    )
```

**Usage:**
```bash
# 1. Launch all training jobs (evals will fail initially)
./tools/run.py experiments.user.lr_sweep.lr_sweep

# 2. Monitor training
./tools/run.py experiments.user.lr_sweep.lr_sweep mode=monitor

# 3. After training completes, manually launch evals with correct checkpoint URIs
# (For v1, this is manual; v2 will auto-resolve dependencies)
```

#### Workflow 2: A/B Test with Analysis

```python
# experiments/user/reward_shaping_ab.py
from metta.experiment.tool import ExperimentTool
from metta.jobs.models import JobSpec

def reward_shaping_ab() -> ExperimentTool:
    """Compare shaped vs unshaped rewards."""

    jobs = [
        JobSpec(
            name="baseline_unshaped",
            module="experiments.recipes.arena.train",
            args={"run": "ab_test.baseline"},
            overrides={"trainer.total_timesteps": 200_000_000},
            gpus=8,
            nodes=2,
        ),
        JobSpec(
            name="shaped_rewards",
            module="experiments.recipes.arena_basic_easy_shaped.train",
            args={"run": "ab_test.shaped"},
            overrides={"trainer.total_timesteps": 200_000_000},
            gpus=8,
            nodes=2,
        ),
    ]

    return ExperimentTool(
        name="reward_shaping_ab",
        jobs=jobs,
    )
```

**Usage:**
```bash
# 1. Launch
./tools/run.py experiments.user.reward_shaping_ab.reward_shaping_ab

# 2. Monitor until complete
./tools/run.py experiments.user.reward_shaping_ab.reward_shaping_ab mode=monitor

# 3. Generate comparison report
./tools/run.py experiments.user.reward_shaping_ab.reward_shaping_ab mode=report
# → Creates experiments/reports/reward_shaping_ab_20251013_1430_report.ipynb
```

#### Workflow 3: Multi-Stage Experiment

For experiments that need sequential stages (train → eval → retrain):

```python
# experiments/user/iterative_training.py
def iterative_training() -> ExperimentTool:
    """Multi-stage experiment: initial train → eval → curriculum adjust → retrain."""

    # For v1, this would be multiple separate experiment launches
    # User would:
    # 1. Launch stage1 experiment
    # 2. Wait for completion
    # 3. Analyze results
    # 4. Launch stage2 experiment with adjusted params

    # In v2, we could add dependency support to automate this

    jobs = [
        JobSpec(name="stage1_train", ...),
        JobSpec(name="stage1_eval", ...),
    ]

    return ExperimentTool(name="iterative_training_stage1", jobs=jobs)
```

---

## Implementation Guide

### Step-by-Step Implementation Order

#### Phase 1: Shared Infrastructure

1. **Create `metta/jobs/models.py`**
   - Implement `JobSpec` dataclass
   - Add serialization methods (`to_dict`, `from_dict`)
   - Add `to_remote_job_args()` method

2. **Move job runner to `metta/jobs/runner.py`**
   - Copy `RemoteJob` and `LocalJob` from `devops/job_runner.py`
   - Add back-compat imports in `devops/job_runner.py`:
     ```python
     # Back-compat: re-export from new location
     from metta.jobs.runner import RemoteJob, LocalJob, JobResult
     ```

3. **Create `metta/jobs/monitor.py`**
   - Extract `poll_job_status()` helper (uses `sky jobs queue`)
   - Add `format_job_table()` (can be simple for v1, enhance later)

4. **Create `metta/jobs/metrics.py`**
   - Extract `extract_wandb_info_from_logs()` from release system
   - Returns dict with `url`, `run_id`, `checkpoint_uri`

#### Phase 2: Experiment Core

5. **Create `metta/experiment/state.py`**
   - Implement `JobState` dataclass
   - Implement `ExperimentState` dataclass
   - Add `save()` and `load()` methods

6. **Create `metta/experiment/tool.py`**
   - Implement `ExperimentTool` class
   - Add `invoke()` method with mode routing
   - Add `_resolve_instance_id()` helper

7. **Create `metta/experiment/launcher.py`**
   - Implement `ExperimentLauncher` class
   - Add `_launch_single_job()` method
   - Handle state updates

8. **Create `metta/experiment/monitor.py`**
   - Implement `ExperimentMonitor` class
   - Add `run()` method for live monitoring
   - Add `attach()` method for log streaming

9. **Create `metta/experiment/manager.py`**
   - Implement `ExperimentManager` class
   - Add `cancel_all()` method

#### Phase 3: Tool Registration & Examples

10. **Create `metta/tools/experiment.py`**
    ```python
    # Just a thin wrapper for tool registration
    from metta.experiment.tool import ExperimentTool

    __all__ = ["ExperimentTool"]
    ```

11. **Update `metta/common/tool/tool_registry.py`**
    ```python
    from metta.tools.experiment import ExperimentTool

    tool_registry.register(ExperimentTool)
    ```

12. **Create `experiments/user/example_experiment.py`**
    - Working example showing basic usage
    - Include comments explaining each part

13. **Update `.gitignore`**
    ```
    experiments/state/
    experiments/logs/
    experiments/reports/
    ```

### Key Implementation Details

#### Handling WandB Groups

All jobs in an experiment should share a WandB group for easy filtering:

```python
# In ExperimentLauncher._launch()
for job in self.jobs:
    if "group" not in job.args:
        job.args["group"] = self.instance_id
```

#### Job ID Tracking

Save job_id to state as soon as job is submitted:

```python
# In ExperimentLauncher._launch_single_job()
job.submit()
if job._job_id:
    state.update_job_status(
        job_spec.name,
        job_id=str(job._job_id),
        status="running",
    )
```

#### Artifact Extraction

Extract WandB info from logs after job completes:

```python
# In ExperimentMonitor._refresh_and_display()
if job_state.status == "completed" and job_state.logs_path:
    wandb_info = extract_wandb_info_from_logs(job_state.logs_path)
    if wandb_info:
        state.update_job_status(
            job_name,
            wandb_url=wandb_info["url"],
            checkpoint_uri=wandb_info["checkpoint_uri"],
        )
```

#### Error Handling

Always update state even on failures:

```python
try:
    job.submit()
    # ...
except Exception as e:
    print(f"✗ Launch failed: {e}")
    state.update_job_status(job_spec.name, status="failed")
    return False
```

---

## Code Reuse Strategy

### From devops/job_runner.py

**Reuse as-is:**
- `RemoteJob` class (move to `metta/jobs/runner.py`)
- `LocalJob` class (move to `metta/jobs/runner.py`)
- `JobResult` dataclass (already perfect)

**Back-compat strategy:**
```python
# devops/job_runner.py (after moving code)
"""Job runner - back-compat imports."""
from metta.jobs.runner import RemoteJob, LocalJob, JobResult, Job

__all__ = ["RemoteJob", "LocalJob", "JobResult", "Job"]
```

### From devops/stable/metrics.py

**Extract and generalize:**
```python
# metta/jobs/metrics.py
import re
from pathlib import Path

_WANDB_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s]+)")

def extract_wandb_info_from_logs(log_path: str) -> dict[str, str] | None:
    """Extract WandB run info from log file.

    Returns:
        {"url": "https://...", "run_id": "abc123", "checkpoint_uri": "wandb://run/abc123"}
        or None if not found
    """
    if not Path(log_path).exists():
        return None

    log_text = Path(log_path).read_text(errors="ignore")
    match = _WANDB_URL_RE.search(log_text)

    if not match:
        return None

    entity, project, run_id = match.groups()

    return {
        "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "run_id": run_id,
        "checkpoint_uri": f"wandb://run/{run_id}",
    }
```

### From metta/adaptive/utils.py

**Reuse table formatting:**
```python
# metta/jobs/monitor.py
def format_job_table(jobs: list[JobState]) -> str:
    """Format job states as a table.

    For v1, keep simple. Can enhance later with adaptive's full formatting.
    """
    # Simple implementation for v1
    lines = []
    lines.append(f"{'Job Name':<30} {'Status':<15} {'Job ID':<10}")
    lines.append(f"{'─'*30} {'─'*15} {'─'*10}")

    for job in jobs:
        lines.append(f"{job.name:<30} {job.status:<15} {job.job_id or '-':<10}")

    return "\n".join(lines)
```

### Skypilot Status Polling

**New utility:**
```python
# metta/jobs/monitor.py
def poll_job_status(job_id: str) -> str | None:
    """Poll Skypilot for job status.

    Returns:
        Status string: "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"
        or None if job not found
    """
    try:
        import sky

        job_statuses = sky.jobs.queue(refresh=True)

        for job in job_statuses:
            if str(job.job_id) == job_id:
                return job.status

        return None

    except Exception:
        return None
```

---

## Summary: Files and Responsibilities

### New Files (14 total)

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `metta/jobs/__init__.py` | Package init | 10 |
| `metta/jobs/models.py` | JobSpec dataclass | 150 |
| `metta/jobs/runner.py` | RemoteJob, LocalJob (moved) | 500 |
| `metta/jobs/state.py` | StateStore protocol | 50 |
| `metta/jobs/monitor.py` | Monitoring utilities | 100 |
| `metta/jobs/metrics.py` | WandB extraction | 50 |
| `metta/experiment/__init__.py` | Package init | 10 |
| `metta/experiment/tool.py` | ExperimentTool class | 200 |
| `metta/experiment/state.py` | State management | 200 |
| `metta/experiment/launcher.py` | Job launching | 250 |
| `metta/experiment/monitor.py` | Live monitoring | 200 |
| `metta/experiment/manager.py` | Cancel/management | 100 |
| `metta/tools/experiment.py` | Tool registration | 10 |
| `experiments/user/example_experiment.py` | Example | 100 |

**Total: ~2,000 lines of new code**

### Modified Files (2 total)

| File | Change | Lines |
|------|--------|-------|
| `devops/job_runner.py` | Add back-compat imports | +5 |
| `.gitignore` | Add experiments/ dirs | +3 |

---

## Quick Reference: Common Patterns

### Pattern 1: Define Simple Experiment

```python
from metta.experiment.tool import ExperimentTool
from metta.jobs.models import JobSpec

def my_experiment() -> ExperimentTool:
    jobs = [
        JobSpec(
            name="job1",
            module="experiments.recipes.arena.train",
            args={"run": "experiment.job1"},
            overrides={"trainer.total_timesteps": 1000000},
        ),
    ]
    return ExperimentTool(name="my_experiment", jobs=jobs)
```

### Pattern 2: Programmatic Job Generation

```python
def param_sweep() -> ExperimentTool:
    jobs = []
    for param_value in [0.1, 0.3, 0.5]:
        jobs.append(JobSpec(
            name=f"param_{param_value}",
            module="experiments.recipes.arena.train",
            args={"run": f"sweep.{param_value}"},
            overrides={"some.param": param_value},
        ))
    return ExperimentTool(name="param_sweep", jobs=jobs)
```

### Pattern 3: Access State Programmatically

```python
from metta.experiment.state import ExperimentState

# Load state
state = ExperimentState.load("my_experiment_20251013_1430")

# Check job status
for job_name, job_state in state.jobs.items():
    print(f"{job_name}: {job_state.status}")
    if job_state.checkpoint_uri:
        print(f"  Checkpoint: {job_state.checkpoint_uri}")
```

### Pattern 4: Query Skypilot Directly

```python
from metta.jobs.monitor import poll_job_status

status = poll_job_status("12345")
print(f"Job 12345 status: {status}")
```

---

**End of Technical Design Document**

This document provides all necessary information to implement the experiment system from scratch. Implementation should follow the phase order and reuse existing code where indicated.
