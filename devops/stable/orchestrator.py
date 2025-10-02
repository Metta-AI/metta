"""Orchestration layer for running validations.

Single entrypoint: run_validations() takes a list of Validation configs,
runs them (local or remote), extracts metrics, evaluates acceptance criteria,
and persists state.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from devops.job_runner import run_local, run_remote
from devops.stable.acceptance import LogRegexMetrics, MetricsSource, evaluate_thresholds
from devops.stable.models import Artifact, GateResult, Lifecycle, Location, Outcome, ReleaseState, RunResult, Validation


class StateManager:
    """Manages release state persistence."""

    def __init__(self, state_dir: Path = Path("devops/stable/state")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_commit_sha(self) -> Optional[str]:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def create_state(self, version: str, repo_root: str) -> ReleaseState:
        """Create a new release state."""
        return ReleaseState(
            version=version,
            repo_root=repo_root,
            commit_sha=self._get_commit_sha(),
            created_at=datetime.utcnow(),
        )

    def save_state(self, state: ReleaseState) -> Path:
        """Save release state to JSON file."""
        filename = f"{state.version}.json"
        path = self.state_dir / filename

        with open(path, "w") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2, default=str)

        return path

    def load_state(self, version: str) -> Optional[ReleaseState]:
        """Load release state from JSON file."""
        filename = f"{version}.json"
        path = self.state_dir / filename

        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)
            return ReleaseState.model_validate(data)


def run_validations(
    version: str,
    validations: list[Validation],
    state_manager: Optional[StateManager] = None,
    metrics_source: Optional[MetricsSource] = None,
) -> ReleaseState:
    """Run validations and return final state.

    Args:
        version: Version/name for this validation run
        validations: List of validations to run
        state_manager: State persistence (default: StateManager)
        metrics_source: Metrics extraction source (default: LogRegexMetrics)

    Returns:
        ReleaseState with all validation results
    """
    # Initialize components
    state_manager = state_manager or StateManager()
    metrics_source = metrics_source or LogRegexMetrics()

    # Create or load state
    state = state_manager.load_state(version)
    if state is None:
        state = state_manager.create_state(version, ".")

    def run_one(v: Validation) -> RunResult:
        """Run a single validation."""
        # Check if already completed
        existing = state.get_validation_result(v.name)
        if existing and existing.lifecycle == Lifecycle.COMPLETED:
            print(f"  Skipping {v.name} - already completed")
            return existing

        # Create run result
        result = RunResult(name=v.name, location=v.location, lifecycle=Lifecycle.PENDING).mark_started()

        try:
            # Launch job
            print(f"\n  Running {v.location} validation: {v.name}")

            if v.location == Location.LOCAL:
                cmd = ["uv", "run", "./tools/run.py", v.module, *v.args]
                job = run_local(name=v.name, cmd=cmd, timeout_s=v.timeout_s, log_dir="devops/stable/logs/local")
                logs_text = job.get_logs()
                result.exit_code = job.exit_code
                result.logs_path = job.logs_path
                result.external_id = None
            else:
                job = run_remote(
                    name=v.name,
                    module=v.module,
                    args=v.args,
                    timeout_s=v.timeout_s,
                    log_dir="devops/stable/logs/remote",
                )
                result.exit_code = job.wait(timeout_s=v.timeout_s)
                logs_text = job.get_logs()
                result.logs_path = job.logs_path
                result.external_id = job.job_id

            # Extract metrics if job succeeded
            if result.exit_code == 0:
                result.metrics = metrics_source.extract(result, logs_text)

                # Evaluate inline acceptance criteria
                if v.acceptance:
                    outcome, failed = evaluate_thresholds(result.metrics, v.acceptance)
                    result = result.mark_completed(outcome, result.exit_code)

                    if outcome == Outcome.PASSED:
                        print(f"  ✅ {v.name} PASSED")
                        if result.metrics:
                            print(f"     Metrics: {result.metrics}")
                    else:
                        print(f"  ❌ {v.name} FAILED acceptance criteria")
                        print(f"     Metrics: {result.metrics}")
                        for check in failed:
                            print(f"     Failed: {check.note}")
                else:
                    # No acceptance criteria - just mark as passed
                    result = result.mark_completed(Outcome.PASSED, result.exit_code)
                    print(f"  ✅ {v.name} PASSED (no acceptance criteria)")
            else:
                # Non-zero exit code
                result = result.mark_failed(f"Exit code {result.exit_code}", result.exit_code)
                print(f"  ❌ {v.name} FAILED - exit code {result.exit_code}")

            # Always attach logs
            if result.logs_path:
                result.artifacts.append(Artifact(name=f"{v.name}.log", uri=result.logs_path))

            return result

        except Exception as e:
            # Handle errors
            result = result.mark_failed(f"Exception: {str(e)}", exit_code=1)
            print(f"  ❌ {v.name} FAILED - {str(e)}")
            return result

    # Run validations in parallel
    with ThreadPoolExecutor(max_workers=min(4, len(validations))) as pool:
        futures = {pool.submit(run_one, v): v for v in validations}
        for fut in as_completed(futures):
            result = fut.result()
            state.add_validation_result(result)
            state_manager.save_state(state)  # Persist incrementally

    return state


def record_workflow_gate(state: ReleaseState, note: str = "") -> None:
    """Record a workflow gate result in the state."""
    summary = state.validation_summary
    outcome = Outcome.PASSED if state.all_validations_passed else Outcome.FAILED
    gate = GateResult(
        name="workflow",
        outcome=outcome,
        notes=f"{note} Passed={summary['passed']} Failed={summary['failed']}",
        artifacts=[Artifact(name="state.json", uri=f"devops/stable/state/{state.version}.json", type="json")],
    )
    state.add_gate_result(gate)


def print_validation_summary(state: ReleaseState) -> None:
    """Print a summary of validation results."""
    print("\n" + "=" * 80)
    print(f"Validation Summary for {state.version}")
    print("=" * 80)

    summary = state.validation_summary
    print("\nResults:")
    print(f"  ✅ Passed:  {summary['passed']}")
    print(f"  ❌ Failed:  {summary['failed']}")
    print(f"  ⏸️  Skipped: {summary['skipped']}")
    if summary["running"] > 0:
        print(f"  🔄 Running: {summary['running']}")

    print("\nDetailed Results:")
    for result in state.validations.values():
        status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(
            result.outcome.value if result.outcome else "", "❓"
        )
        print(f"  {status_icon} {result.name:24} loc={result.location:6} exit={result.exit_code:>3}")
        if result.metrics:
            metrics_str = "  ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            print(f"       Metrics: {metrics_str}")
        if result.logs_path:
            print(f"       Logs: {result.logs_path}")
        if result.external_id:
            print(f"       Job ID: {result.external_id}")

    print("=" * 80)
