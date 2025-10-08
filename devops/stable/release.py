#!/usr/bin/env -S uv run
"""Stable Release System (lean, single-file)

Usage:
  ./devops/stable/release.py                    # Run full release flow
  ./devops/stable/release.py --prepare-branch   # Create staging branch
  ./devops/stable/release.py --bugs             # Check bug status
  ./devops/stable/release.py --workflow test    # Run TEST workflows
  ./devops/stable/release.py --workflow train   # Run TRAIN workflows
  ./devops/stable/release.py --summary          # Show validation summary
  ./devops/stable/release.py --release          # Create release tag

Auto-resume:
  By default, continues the most recent in-progress release.
  Use --new to force start a new release.
  Use --version X to use a specific version.

Workflow filtering:
  --workflow ci                    # Run CI workflow (metta ci - tests + linting)
  --workflow train                 # Run all TRAIN workflows (local + remote)
  --workflow train_local           # Run local training smoke tests
  --workflow train_remote          # Run remote single-GPU training
  --workflow train_remote_multigpu # Run remote multi-GPU training
  --workflow play                  # Run all PLAY workflows (interactive testing)
  --workflow metta_ci              # Run specific validation by name
  --workflow arena_local_smoke     # Run specific validation by name

Examples:
  ./devops/stable/release.py                               # Full release (auto-continue)
  ./devops/stable/release.py --new                         # Full release (force new)
  ./devops/stable/release.py --workflow ci                 # Just run CI (tests + linting)
  ./devops/stable/release.py --workflow train_local        # Run local smoke test
  ./devops/stable/release.py --workflow train_remote       # Run single-GPU remote training
  ./devops/stable/release.py --workflow train --new        # Run all train workflows (force new)
  ./devops/stable/release.py --version 2025.10.07-test1    # Use specific version
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Optional

import gitta as git
from devops.job_runner import run_local, run_remote
from metta.common.util.text_styles import bold, cyan, green, red

# ============================================================================
# Data Models
# ============================================================================

Location = Literal["local", "remote"]
Outcome = Literal["passed", "failed", "skipped", "inconclusive"]


class WorkflowType(StrEnum):
    """Types of workflow validations."""

    CI = "ci"  # Run metta ci locally (tests + linting)
    TRAIN_LOCAL = "train_local"  # Local smoke test with manual validation
    TRAIN_REMOTE = "train_remote"  # Remote single-GPU training with strict thresholds
    TRAIN_REMOTE_MULTIGPU = "train_remote_multigpu"  # Remote multi-GPU/multi-node training
    PLAY = "play"  # Run play recipe with manual verification


@dataclass
class ThresholdCheck:
    """A single acceptance criterion (e.g., SPS >= 40000)."""

    key: str
    op: Literal[">=", ">", "<=", "<", "==", "!="] = ">="
    expected: float = 0.0
    actual: Optional[float] = None
    passed: Optional[bool] = None
    note: Optional[str] = None


@dataclass
class Validation:
    """Configuration for a validation run."""

    name: str
    workflow_type: WorkflowType
    module: str
    location: Location
    args: list[str] = field(default_factory=list)
    timeout_s: int = 900
    acceptance: list[ThresholdCheck] = field(default_factory=list)


@dataclass
class ValidationResult:
    """State record for a validation run (aggregates job execution + validation outcome)."""

    name: str
    workflow_type: WorkflowType
    location: Location
    started_at: str
    ended_at: Optional[str] = None
    exit_code: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    logs_path: Optional[str] = None
    job_id: Optional[str] = None
    outcome: Optional[Outcome] = None
    error: Optional[str] = None

    def complete(
        self,
        outcome: Outcome,
        exit_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark this validation as completed."""
        if exit_code is not None:
            self.exit_code = exit_code
        self.outcome = outcome
        self.ended_at = datetime.utcnow().isoformat(timespec="seconds")
        if error:
            self.error = error


@dataclass
class ReleaseState:
    """State of a release qualification run."""

    version: str
    created_at: str
    commit_sha: Optional[str] = None
    validations: dict[str, ValidationResult] = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    released: bool = False


# ============================================================================
# State Persistence
# ============================================================================

STATE_DIR = Path("devops/stable/state")
LOG_DIR_LOCAL = Path("devops/stable/logs/local")
LOG_DIR_REMOTE = Path("devops/stable/logs/remote")

# Contacts
CONTACTS = [
    "Release Manager: @Robb",
    "Technical Lead: @Jack Heart",
    "Bug Triage: @Nishad Singh",
]

# Create directories
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
LOG_DIR_REMOTE.mkdir(parents=True, exist_ok=True)


def _get_commit_sha() -> Optional[str]:
    """Get current git commit SHA."""
    try:
        return git.get_current_commit()
    except git.GitError:
        return None


def _get_git_log_since_stable() -> str:
    """Get git log since the last stable release tag."""
    try:
        # Find the latest stable tag
        last_tag = git.run_git("describe", "--tags", "--abbrev=0", "--match=v*", "stable")

        # Get git log from that tag to HEAD
        return git.run_git("log", f"{last_tag}..HEAD", "--oneline")
    except git.GitError:
        # If no stable tag found, just return recent commits
        try:
            return git.run_git("log", "--oneline", "-20")
        except git.GitError:
            return "Unable to retrieve git log"


def _recursive_asdict(obj):
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _recursive_asdict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: _recursive_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_recursive_asdict(item) for item in obj]
    return obj


def load_state(version: str) -> Optional[ReleaseState]:
    """Load release state from JSON file.

    Args:
        version: Version string (with or without 'release_' prefix)
    """
    # Ensure version has release_ prefix for filename
    if not version.startswith("release_"):
        version = f"release_{version}"

    path = STATE_DIR / f"{version}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())

        # Reconstruct nested dataclasses
        validations = {}
        for name, val_data in data.get("validations", {}).items():
            if not isinstance(val_data, dict):
                raise ValueError(f"Invalid validation data for {name}")
            validations[name] = ValidationResult(**val_data)

        state = ReleaseState(
            version=data["version"],
            created_at=data["created_at"],
            commit_sha=data.get("commit_sha"),
            validations=validations,
            gates=data.get("gates", []),
            released=data.get("released", False),
        )
        return state
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Failed to load state from {path}: {e}")
        return None


def get_most_recent_state() -> Optional[tuple[str, ReleaseState]]:
    """Get the most recent release state.

    Returns:
        Tuple of (version, state) or None if no state files exist
    """
    state_files = sorted(STATE_DIR.glob("release_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not state_files:
        return None

    # Try to load the most recent state
    most_recent = state_files[0]
    version = most_recent.stem.replace("release_", "")
    state = load_state(version)

    if state:
        return (version, state)
    return None


def save_state(state: ReleaseState) -> Path:
    """Save release state to JSON file with atomic write.

    Uses atomic write pattern to prevent corruption from concurrent writes.
    """
    version = state.version
    # Ensure version has release_ prefix for filename
    if not version.startswith("release_"):
        version = f"release_{version}"

    path = STATE_DIR / f"{version}.json"
    temp_path = path.with_suffix(".json.tmp")

    serialized = _recursive_asdict(state)

    # Write to temp file first
    temp_path.write_text(json.dumps(serialized, indent=2))

    # Atomic rename (on POSIX systems this is atomic)
    temp_path.replace(path)

    return path


def update_validation_result(version: str, validation_name: str, result: ValidationResult) -> None:
    """Atomically update a single validation result in the state file.

    This prevents race conditions when multiple processes are running different validations.
    Uses a simple retry loop with file locking via exclusive creation.
    """
    # Ensure version has release_ prefix for lock filename
    lock_version = version if version.startswith("release_") else f"release_{version}"
    lock_path = STATE_DIR / f"{lock_version}.lock"
    max_retries = 10
    retry_delay = 0.5  # seconds

    for attempt in range(max_retries):
        try:
            # Try to acquire lock by creating lock file exclusively
            # This fails if file already exists (another process has the lock)
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

            try:
                # We have the lock - reload state, update, and save
                state = load_state(version)
                if not state:
                    # State was deleted or doesn't exist - this is unexpected
                    print(f"Warning: State file not found during update for {validation_name}")
                    return

                state.validations[validation_name] = result
                save_state(state)

            finally:
                # Release lock
                os.close(lock_fd)
                lock_path.unlink()

            return  # Success

        except FileExistsError:
            # Lock is held by another process - retry after delay
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # Give up after max retries
                print(f"Warning: Could not acquire lock after {max_retries} attempts for {validation_name}")
                # Fall back to direct save (may cause race condition)
                state = load_state(version)
                if state:
                    state.validations[validation_name] = result
                    save_state(state)


# ============================================================================
# Metrics Extraction & Acceptance Criteria
# ============================================================================

# Regex patterns for extracting metrics from logs
_SPS_RE = re.compile(r"\bSPS[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_KSPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ksps", re.IGNORECASE)  # Match "87.75 ksps"
_EVAL_RATE_RE = re.compile(r"\beval[_\s-]?success[_\s-]?rate[:=]\s*(0?\.\d+|1(?:\.0)?)", re.IGNORECASE)


def extract_metrics(log_text: str) -> dict[str, float]:
    """Extract metrics from log text using regex patterns.

    Supports:
    - SPS (samples per second) - max and last value
    - KSPS (kilosamples per second) - converted to SPS
    - Eval success rate
    """
    metrics: dict[str, float] = {}

    # Extract SPS values (plain format: "SPS: 1234" or "SPS=1234")
    sps_matches = [float(x) for x in _SPS_RE.findall(log_text)]

    # Extract KSPS values (progress logger format: "87.75 ksps")
    ksps_matches = [float(x) * 1000 for x in _KSPS_RE.findall(log_text)]

    # Combine both formats
    all_sps = sps_matches + ksps_matches
    if all_sps:
        metrics["sps_max"] = max(all_sps)
        metrics["sps_last"] = all_sps[-1]

    # Extract eval success rate
    eval_matches = _EVAL_RATE_RE.findall(log_text)
    if eval_matches:
        metrics["eval_success_rate"] = float(eval_matches[-1])

    return metrics


# Threshold operators
_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def evaluate_thresholds(
    metrics: dict[str, float], checks: list[ThresholdCheck]
) -> tuple[Outcome, list[ThresholdCheck]]:
    """Evaluate metrics against threshold checks.

    Returns:
        Tuple of (outcome, failed_checks)
    """
    failed: list[ThresholdCheck] = []
    for check in checks:
        # Missing metrics are hard failures
        if check.key not in metrics:
            check.actual = None
            check.passed = False
            check.note = "metric missing"
            failed.append(check)
            continue

        check.actual = metrics[check.key]
        passed = _OPS[check.op](check.actual, check.expected)
        check.passed = bool(passed)

        if not passed:
            check.note = f"expected {check.op} {check.expected}, saw {check.actual}"
            failed.append(check)

    return ("passed" if not failed else "failed", failed)


# ============================================================================
# Validation Execution
# ============================================================================


def run_validation(
    state: ReleaseState, validation: Validation, cluster_config: Optional[dict] = None
) -> ValidationResult:
    """Run a single validation based on workflow type.

    Args:
        state: Current release state
        validation: Validation configuration
        cluster_config: Optional cluster configuration (nodes, gpus, etc.)

    Returns:
        ValidationResult with outcome and metrics
    """
    # Skip if already passed or skipped (allow retrying failed validations)
    if validation.name in state.validations:
        existing = state.validations[validation.name]
        if existing.outcome in ("passed", "skipped"):
            print(f"  ⏭️  {validation.name} - already completed ({existing.outcome})")
            return existing
        elif existing.outcome == "failed":
            print(f"  🔄 {validation.name} - retrying previous failure...")

    # Create new result record (for state persistence)
    result = ValidationResult(
        name=validation.name,
        workflow_type=validation.workflow_type,
        location=validation.location,
        started_at=datetime.utcnow().isoformat(timespec="seconds"),
    )

    try:
        print(f"  🔄 {validation.name} [{validation.workflow_type.value}] - starting...")

        # Handle different workflow types
        if validation.workflow_type == WorkflowType.CI:
            # Run metta ci locally via job_runner
            cmd = ["metta", "ci"]
            job = run_local(
                name=validation.name,
                cmd=cmd,
                timeout_s=validation.timeout_s,
                log_dir=str(LOG_DIR_LOCAL),
                stream_output=True,
            )
            # Extract data from job_runner result for state persistence
            log_text = job.get_logs()
            result.exit_code = job.exit_code
            result.logs_path = job.logs_path

        elif validation.workflow_type == WorkflowType.PLAY:
            # Run play recipe locally via job_runner
            cmd = ["uv", "run", "./tools/run.py", validation.module, *validation.args]
            print("     Launching play window for manual verification...")
            print(f"     Command: {' '.join(cmd)}")

            job = run_local(
                name=validation.name,
                cmd=cmd,
                timeout_s=validation.timeout_s,
                log_dir=str(LOG_DIR_LOCAL),
                stream_output=True,
            )
            # Extract data from job_runner result for state persistence
            result.exit_code = job.exit_code
            result.logs_path = job.logs_path
            log_text = job.get_logs()

            # Always require manual confirmation for PLAY workflows
            if result.exit_code == 0:
                print("     Play window launched successfully")
                if not _get_user_confirmation(f"Did the play workflow '{validation.name}' work correctly?"):
                    result.complete("failed", 1, error="User indicated play workflow failed")
                    print(f"  ❌ {validation.name} - User confirmed FAILURE")
                    return result

        elif validation.workflow_type in (
            WorkflowType.TRAIN_LOCAL,
            WorkflowType.TRAIN_REMOTE,
            WorkflowType.TRAIN_REMOTE_MULTIGPU,
        ):
            # Run training job via job_runner (local or remote)
            if validation.location == "local":
                cmd = ["uv", "run", "./tools/run.py", validation.module, *validation.args]
                job = run_local(
                    name=validation.name,
                    cmd=cmd,
                    timeout_s=validation.timeout_s,
                    log_dir=str(LOG_DIR_LOCAL),
                    stream_output=True,
                )
                # Extract data from LocalJobResult for state persistence
                log_text = job.get_logs()
                result.exit_code = job.exit_code
                result.logs_path = job.logs_path

                # Extract W&B URL and ask for manual confirmation
                wandb_url_match = re.search(r"https://wandb\.ai/[^\s]+", log_text)
                if wandb_url_match:
                    wandb_url = wandb_url_match.group(0)
                    print(f"\n     📊 W&B Run: {wandb_url}")
                    print("     Please verify:")
                    print("       - SPS (samples per second) looks reasonable for local")
                    print("       - Training completed successfully")
                    print("       - No errors in logs")
                    if not _get_user_confirmation(f"Do the metrics look good for '{validation.name}'?"):
                        result.complete("failed", 1, error="User indicated metrics look bad")
                        print(f"  ❌ {validation.name} - User confirmed FAILURE")
                        return result
            else:
                # Build base_args for SkyPilot
                config = cluster_config or {}
                nodes = config.get("nodes", 1)
                gpus = config.get("gpus", 4)
                cloud = config.get("cloud", None)
                region = config.get("region", None)

                base_args = ["--no-spot", f"--gpus={gpus}", "--nodes", str(nodes)]
                if cloud:
                    base_args.extend(["--cloud", cloud])
                if region:
                    base_args.extend(["--region", region])

                print(f"     Cluster config: {nodes} nodes × {gpus} GPUs = {nodes * gpus} total GPUs")

                # Check if we should resume existing job
                existing_job_id = None
                if validation.name in state.validations:
                    existing_result = state.validations[validation.name]
                    if existing_result.job_id and existing_result.outcome not in ("passed", "failed"):
                        # Check if job still exists
                        from devops.skypilot.utils.job_helpers import check_job_statuses

                        job_status = check_job_statuses([int(existing_result.job_id)])
                        if int(existing_result.job_id) in job_status:
                            status = job_status[int(existing_result.job_id)].get("status", "UNKNOWN")
                            if status not in ("UNKNOWN", "ERROR"):
                                existing_job_id = existing_result.job_id
                                print(f"     📋 Resuming existing job {existing_job_id} (status: {status})")

                job = run_remote(
                    name=validation.name,
                    module=validation.module,
                    args=validation.args,
                    timeout_s=validation.timeout_s,
                    log_dir=str(LOG_DIR_REMOTE),
                    base_args=base_args,
                    job_id=existing_job_id,
                )

                # Save job_id to state immediately after submission
                result.job_id = job.job_id
                # Use atomic update to prevent race conditions with concurrent workflows
                update_validation_result(state.version, validation.name, result)

                result.exit_code = job.wait(timeout_s=validation.timeout_s, stream_output=True)
                # Extract data from RemoteJob for state persistence
                log_text = job.get_logs()
                result.logs_path = job.logs_path

                # Extract W&B URL and ask for manual confirmation
                wandb_url_match = re.search(r"https://wandb\.ai/[^\s]+", log_text)
                if wandb_url_match:
                    wandb_url = wandb_url_match.group(0)
                    print(f"\n     📊 W&B Run: {wandb_url}")
                    print("     Please verify:")
                    print("       - SPS (samples per second) looks reasonable")
                    print("       - Heartbeat is consistent")
                    print("       - No anomalies in training curves")
                    if not _get_user_confirmation(f"Do the metrics look good for '{validation.name}'?"):
                        result.complete("failed", 1, error="User indicated metrics look bad")
                        print(f"  ❌ {validation.name} - User confirmed FAILURE")
                        return result

        # Check exit code
        if result.exit_code == 124:
            result.complete("failed", result.exit_code, error="Timeout exceeded")
            print(red(f"  ❌ {validation.name} - TIMEOUT"))
            return result
        elif result.exit_code != 0:
            result.complete("failed", result.exit_code, error=f"Exit code {result.exit_code}")
            print(f"  ❌ {validation.name} - FAILED (exit {result.exit_code})")
            return result

        # Extract metrics (only for TRAIN workflows)
        if validation.workflow_type in (
            WorkflowType.TRAIN_LOCAL,
            WorkflowType.TRAIN_REMOTE,
            WorkflowType.TRAIN_REMOTE_MULTIGPU,
        ):
            result.metrics = extract_metrics(log_text or "")

        # Evaluate acceptance criteria
        if validation.acceptance:
            outcome, failed_checks = evaluate_thresholds(result.metrics, validation.acceptance)
            result.complete(outcome, result.exit_code)

            if outcome == "passed":
                print(green(f"  ✅ {validation.name} - PASSED"))
                if result.metrics:
                    metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
                    print(f"     Metrics: {metrics_str}")
            else:
                print(f"  ❌ {validation.name} - FAILED acceptance criteria")
                if result.metrics:
                    metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
                    print(f"     Metrics: {metrics_str}")
                for check in failed_checks:
                    print(f"     Failed: {check.note}")
        else:
            # No acceptance criteria - just mark as passed
            result.complete("passed", result.exit_code)
            print(f"  ✅ {validation.name} - PASSED")

        return result

    except subprocess.TimeoutExpired as e:
        result.complete("failed", 124, error=f"Timeout: {e}")
        print(f"  ❌ {validation.name} - TIMEOUT")
        return result
    except Exception as e:
        result.complete("failed", 1, error=f"Unexpected error: {e}")
        print(f"  ❌ {validation.name} - ERROR: {e}")
        return result


# ============================================================================
# Release Plans
# ============================================================================


def get_workflow_tests() -> list[Validation]:
    """Get the validation plan for this release.

    Returns:
        List of validations to run (CI, TRAIN single-GPU, TRAIN multi-GPU, PLAY)
    """
    validations = []

    # 1. CI workflow - run metta ci locally (tests + linting)
    validations.append(
        Validation(
            name="metta_ci",
            workflow_type=WorkflowType.CI,
            module="",  # Not used for CI workflow
            location="local",
            timeout_s=1800,  # 30 minutes
        )
    )

    # 2. TRAIN_LOCAL - local smoke test
    # Use timestamp-based run name to avoid resuming completed runs
    from datetime import datetime

    smoke_run = f"stable.smoke.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    validations.append(
        Validation(
            name="arena_local_smoke",
            workflow_type=WorkflowType.TRAIN_LOCAL,
            module="experiments.recipes.arena_basic_easy_shaped.train",
            location="local",
            args=[f"run={smoke_run}", "trainer.total_timesteps=1000", "wandb.enabled=false"],
            timeout_s=600,
            acceptance=[],  # Manual validation via W&B URL
        )
    )

    # 3. TRAIN_REMOTE - single GPU remote validation
    validations.append(
        Validation(
            name="arena_single_gpu_10m",
            workflow_type=WorkflowType.TRAIN_REMOTE,
            module="experiments.recipes.arena_basic_easy_shaped.train",
            location="remote",
            args=["trainer.total_timesteps=10000000"],
            timeout_s=3600,  # 1 hour
            acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=40000)],
        )
    )

    # 4. TRAIN_REMOTE_MULTIGPU - multi-GPU/multi-node remote validation
    validations.append(
        Validation(
            name="arena_multi_gpu_2b",
            workflow_type=WorkflowType.TRAIN_REMOTE_MULTIGPU,
            module="experiments.recipes.arena_basic_easy_shaped.train",
            location="remote",
            args=["trainer.total_timesteps=2000000000"],  # 2B timesteps
            timeout_s=86400,  # 24 hours
            acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=40000)],
        )
    )

    # 5. PLAY workflow - interactive testing
    validations.append(
        Validation(
            name="arena_play",
            workflow_type=WorkflowType.PLAY,
            module="experiments.recipes.arena.play",
            location="local",
            args=["policy_uri=file://./train_dir/stable.smoke/checkpoints"],  # Use smoke test checkpoint
            timeout_s=600,  # 10 minutes for manual testing
        )
    )

    return validations


# ============================================================================
# Release Steps
# ============================================================================


def _ensure_git_repo() -> None:
    """Ensure we're in a git repository."""
    try:
        git.run_git("rev-parse", "--git-dir")
    except git.GitError:
        print("Not in a git repository. Aborting.")
        sys.exit(1)


def step_prepare_branch(version: str, **_kwargs) -> None:
    """Step 1: Create and push staging branch."""
    _ensure_git_repo()

    branch_name = f"staging/v{version}-rc1"

    print("\n" + "=" * 60)
    print(bold(f"STEP 1: Prepare Staging Branch (v{version})"))
    print("=" * 60 + "\n")

    print(f"Creating branch: {branch_name}")
    try:
        git.run_git("checkout", "-b", branch_name)
        git.run_git("push", "-u", "origin", branch_name)
        print(green(f"✅ Branch {branch_name} created and pushed successfully"))
    except git.GitError as e:
        print(f"Failed to create/push branch: {e}")
        sys.exit(1)


def _check_asana_blockers() -> Optional[bool]:
    """Check for blocking bugs in Asana Active section.

    Returns:
        True if no blockers (clear to ship)
        False if blockers exist
        None if check is inconclusive (Asana unavailable)
    """
    token = os.getenv("ASANA_TOKEN")
    project_id = os.getenv("ASANA_PROJECT_ID")

    if not (token and project_id):
        return None  # Asana not configured

    try:
        import asana
    except ImportError:
        print("Asana library not installed (pip install asana)")
        return None

    try:
        # Initialize Asana client
        config = asana.Configuration()
        config.access_token = token
        client = asana.ApiClient(config)

        # Get user info to verify auth
        users_api = asana.UsersApi(client)
        user = users_api.get_user("me", {})
        print(f"✓ Authenticated as {user.get('name', '?')}")

        # Get project sections
        sections_api = asana.SectionsApi(client)
        sections = sections_api.get_sections_for_project(project_id, {})

        # Find "Active" section
        active_section = next((s for s in sections if s["name"].lower() == "active"), None)
        if not active_section:
            print("No 'Active' section found in Asana project")
            return None

        # Get tasks in Active section
        tasks_api = asana.TasksApi(client)
        tasks = tasks_api.get_tasks_for_section(
            active_section["gid"],
            {"opt_fields": "name,completed,permalink_url"},
        )

        # Filter for incomplete tasks
        open_tasks = [t for t in tasks if not t.get("completed", False)]

        if not open_tasks:
            print(green("✅ No blocking bugs in Asana Active section"))
            return True

        print(f"❌ Found {len(open_tasks)} blocking task(s) in Active:")
        for task in open_tasks[:10]:  # Show first 10
            print(f"  • {task['name']}")
            if task.get("permalink_url"):
                print(f"    {task['permalink_url']}")
        return False

    except Exception as e:
        print(f"Asana API error: {e}")
        return None


def _get_user_confirmation(prompt: str) -> bool:
    """Get yes/no confirmation from user."""
    try:
        response = input(f"{prompt} [y/N] ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def step_bug_check(**_kwargs) -> None:
    """Step 2: Check for blocking bugs."""
    print("\n" + "=" * 60)
    print(bold("STEP 2: Bug Status Check"))
    print("=" * 60 + "\n")

    # Try automated Asana check
    result = _check_asana_blockers()

    if result is True:
        print("✅ Bug check PASSED - clear for release")
        return
    elif result is False:
        print("❌ Bug check FAILED - resolve blocking issues before release")
        sys.exit(1)

    # Asana check inconclusive - fall back to manual
    print("⚠️  Asana automation unavailable or inconclusive")
    print("\nTo enable automated checking:")
    print("  export ASANA_TOKEN='your_personal_access_token'")
    print("  export ASANA_PROJECT_ID='your_project_id'")
    print("\nManual steps required:")
    print("1. Open Asana project for bug tracking")
    print("2. Verify no active/open bugs marked as blockers")
    print("3. Update bug statuses as needed in consultation with bug owners")
    print("")

    if not _get_user_confirmation("Have you completed the bug status check and is it PASSED?"):
        print("❌ Bug check FAILED - user indicated issues remain")
        sys.exit(1)

    print("✅ Bug check PASSED - user confirmed")


def step_workflow_tests(version: str, workflow_filter: Optional[str] = None, **_kwargs) -> None:
    """Step 3: Run validation workflows.

    Args:
        version: Release version
        workflow_filter: Optional filter - can be:
            - Workflow type: "test", "train", "play"
            - Validation name: "metta_test", "arena_local_smoke", etc.
            - None: run all workflows
    """
    print("\n" + "=" * 60)
    print(bold("STEP 3: Workflow Validation"))
    print("=" * 60 + "\n")

    # Load or create state
    state_version = f"release_{version}"
    state = load_state(state_version)
    if not state:
        state = ReleaseState(
            version=state_version,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            commit_sha=_get_commit_sha(),
        )
        # Save initial state to disk so update_validation_result() can reload it
        save_state(state)

    # Get all validations
    all_validations = get_workflow_tests()

    # Filter validations if requested
    if workflow_filter:
        # Check if filter is a workflow type
        workflow_filter_lower = workflow_filter.lower()
        if workflow_filter_lower in ("ci", "train_local", "train_remote", "train_remote_multigpu", "play"):
            # Exact workflow type match
            validations = [v for v in all_validations if v.workflow_type.value == workflow_filter_lower]
            print(f"Running {workflow_filter.upper()} workflows only\n")
        elif workflow_filter_lower == "train":
            # Match all training workflows
            validations = [v for v in all_validations if v.workflow_type.value.startswith("train_")]
            print("Running ALL TRAIN workflows\n")
        else:
            # Filter by validation name
            validations = [v for v in all_validations if v.name == workflow_filter]
            if not validations:
                print(f"Error: Unknown workflow '{workflow_filter}'")
                print(f"Available workflows: {', '.join(v.name for v in all_validations)}")
                sys.exit(1)
            print(f"Running workflow: {workflow_filter}\n")
    else:
        validations = all_validations
        print("Running all workflows\n")

    # Run validations sequentially
    for validation in validations:
        # Determine cluster config based on workflow type
        if validation.workflow_type == WorkflowType.TRAIN_REMOTE_MULTIGPU:
            cluster_config = {"nodes": 4, "gpus": 4}
        elif validation.workflow_type == WorkflowType.TRAIN_REMOTE:
            cluster_config = {"nodes": 1, "gpus": 1}
        else:
            cluster_config = None

        result = run_validation(state, validation, cluster_config=cluster_config)
        # Use atomic update to prevent race conditions with concurrent workflows
        update_validation_result(state.version, validation.name, result)

    # Reload state to get latest results from all concurrent workflows
    state = load_state(state_version) or state

    # Print summary
    passed = sum(1 for r in state.validations.values() if r.outcome == "passed")
    failed = sum(1 for r in state.validations.values() if r.outcome == "failed")

    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print("\nResults:")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")

    print("\nDetailed Results:")
    for result in state.validations.values():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(result.outcome or "", "❓")
        print(f"  {icon} {result.name:24} loc={result.location:6} exit={result.exit_code:>3}")
        if result.metrics:
            metrics_str = "  ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            print(f"       Metrics: {metrics_str}")
        if result.logs_path:
            print(f"       Logs: {result.logs_path}")
        if result.job_id:
            print(f"       Job ID: {result.job_id}")

    print("=" * 80)

    if failed:
        print(f"\nState saved to: {STATE_DIR / f'{state_version}.json'}")
        print("❌ Workflow validation FAILED")
        sys.exit(1)

    print("\n✅ All workflow validations PASSED")
    print(f"State saved to: {STATE_DIR / f'{state_version}.json'}")


def step_summary(version: str, **_kwargs) -> None:
    """Step 4: Print validation summary and release notes template."""
    print("\n" + "=" * 60)
    print(bold("STEP 4: Release Summary"))
    print("=" * 60 + "\n")

    # Load state to extract metrics
    state_version = f"release_{version}"
    state = load_state(state_version)

    if not state:
        print(f"No state found for version {version}")
        print("Run --step workflow-tests first")
        sys.exit(1)

    # Extract training metrics from TRAIN validations
    training_metrics = {}
    training_job_id = None
    for name, result in state.validations.items():
        if "train" in name.lower() and result.metrics:
            training_metrics.update(result.metrics)
            if result.job_id:
                training_job_id = result.job_id

    # Format metrics for display
    sps_max = training_metrics.get("sps_max", "N/A")
    sps_last = training_metrics.get("sps_last", "N/A")

    # Get git log since last stable release
    git_log = _get_git_log_since_stable()

    # Print validation summary
    print("Validation Results:")
    for name, result in state.validations.items():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(result.outcome or "", "❓")
        print(f"  {icon} {name}")
        if result.metrics:
            metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            print(f"       Metrics: {metrics_str}")

    # Print release notes template
    print("\n" + "=" * 60)
    print("Release Notes Template")
    print("=" * 60 + "\n")
    print(f"## Version {version}")
    print("")
    print("### Changes Since Last Stable Release")
    print("")
    if git_log:
        for line in git_log.split("\n"):
            print(f"- {line}")
    else:
        print("- <No commits found>")
    print("")
    print("### Known Issues")
    print("")
    print("<Add notes from bug-check step>")
    print("")
    print("### Training Job Links")
    print("")
    if training_job_id:
        print(f"- SkyPilot Job ID: {training_job_id}")
        print(f"- View logs: sky logs {training_job_id}")
    else:
        print("- No remote training jobs")
    print("")
    print("### Key Metrics")
    print("")
    print(f"- Training throughput (SPS max): {sps_max}")
    print(f"- Training throughput (SPS last): {sps_last}")
    print("")


def step_release(version: str, **_kwargs) -> None:
    """Step 5: Automatically create release tag and release notes."""
    print("\n" + "=" * 60)
    print(bold("STEP 5: Create Release"))
    print("=" * 60 + "\n")

    _ensure_git_repo()

    # Load state to get validation results
    state_version = f"release_{version}"
    state = load_state(state_version)

    if not state:
        print(f"No state found for version {version}")
        print("Run --step workflow-tests first")
        sys.exit(1)

    # Verify all validations passed
    failed = [name for name, result in state.validations.items() if result.outcome == "failed"]
    if failed:
        print("Cannot release with failed validations:")
        for name in failed:
            print(f"  ❌ {name}")
        sys.exit(1)

    # Extract metrics for release notes
    training_metrics = {}
    training_job_id = None
    for name, result in state.validations.items():
        if "train" in name.lower() and result.metrics:
            training_metrics.update(result.metrics)
            if result.job_id:
                training_job_id = result.job_id

    sps_max = training_metrics.get("sps_max", "N/A")
    sps_last = training_metrics.get("sps_last", "N/A")
    git_log = _get_git_log_since_stable()

    # Create release notes
    release_notes_dir = Path("devops/stable/release-notes")
    release_notes_dir.mkdir(parents=True, exist_ok=True)
    release_notes_path = release_notes_dir / f"v{version}.md"

    release_notes_content = f"""# Release Notes - Version {version}

## Validation Summary

"""
    for name, result in state.validations.items():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(result.outcome or "", "❓")
        release_notes_content += f"- {icon} {name}\n"
        if result.metrics:
            metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            release_notes_content += f"  - Metrics: {metrics_str}\n"

    release_notes_content += f"""
## Changes Since Last Stable Release

{git_log if git_log else "No commits found"}

## Training Job Links

"""
    if training_job_id:
        release_notes_content += f"- SkyPilot Job ID: {training_job_id}\n"
        release_notes_content += f"- View logs: `sky logs {training_job_id}`\n"
    else:
        release_notes_content += "- No remote training jobs\n"

    release_notes_content += f"""
## Key Metrics

- Training throughput (SPS max): {sps_max}
- Training throughput (SPS last): {sps_last}
"""

    # Write release notes
    release_notes_path.write_text(release_notes_content)
    print(f"✅ Created release notes: {release_notes_path}")

    # Create git tag
    tag_name = f"v{version}"
    try:
        # Check if tag already exists
        existing = git.run_git("tag", "-l", tag_name)
        if existing.strip():
            print(f"Tag {tag_name} already exists")
            sys.exit(1)

        # Create annotated tag
        git.run_git("tag", "-a", tag_name, "-m", f"Release version {version}")
        print(f"✅ Created git tag: {tag_name}")

        # Push tag
        git.run_git("push", "origin", tag_name)
        print(f"✅ Pushed tag to origin: {tag_name}")

    except git.GitError as e:
        print(f"Failed to create/push tag: {e}")
        sys.exit(1)

    # Mark state as released
    state.released = True
    save_state(state)

    print("\n" + "=" * 60)
    print("Release Complete!")
    print("=" * 60)
    print(f"\nRelease notes: {release_notes_path}")
    print(f"Git tag: {tag_name}")
    print("\nNext steps:")
    print("  1. Create GitHub release from tag")
    print("  2. Run --step announce to notify team")


def step_announce(version: str, **_kwargs) -> None:
    """Step 6: Print announcement instructions."""
    print("\n" + "=" * 60)
    print(bold("STEP 6: Announce"))
    print("=" * 60 + "\n")

    print("Post release completion to Discord in #eng-process")
    print("\nSuggested message:")
    print(f"Released stable version v{version}")
    print(f"Release notes: devops/stable/release-notes/v{version}.md")


# ============================================================================
# CLI
# ============================================================================


class Step:
    """Step identifiers for the release process."""

    PREPARE = "prepare-branch"
    BUG = "bug-check"
    TESTS = "workflow-tests"
    SUMMARY = "summary"
    RELEASE = "release"
    ANNOUNCE = "announce"


def generate_version() -> str:
    """Generate version string from current date/time.

    Format: YYYY.MM.DD-HHMM (e.g., 2025.10.07-1430)
    """
    return datetime.now().strftime("%Y.%m.%d-%H%M")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Stable Release System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run full release flow
  %(prog)s --new                     # Start new release
  %(prog)s --prepare-branch          # Only create staging branch
  %(prog)s --bugs                    # Only check bugs
  %(prog)s --workflow test           # Run TEST workflows only
  %(prog)s --workflow train          # Run TRAIN workflows only
  %(prog)s --workflow metta_test     # Run specific validation
  %(prog)s --summary                 # Show validation summary
  %(prog)s --release                 # Create release tag
""",
    )

    parser.add_argument(
        "--version",
        help="Version number (overrides auto-continue behavior)",
        default=None,
    )

    parser.add_argument(
        "--new",
        action="store_true",
        help="Force start a new release (ignore existing in-progress state)",
    )

    # Step flags
    parser.add_argument(
        "--prepare-branch",
        action="store_true",
        help="Create and push staging branch",
    )

    parser.add_argument(
        "--bugs",
        action="store_true",
        help="Check bug status in Asana",
    )

    parser.add_argument(
        "--workflow",
        metavar="FILTER",
        help=(
            "Run workflow validations (ci|train|train_local|train_remote|train_remote_multigpu|play|<validation_name>)"
        ),
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show validation summary and release notes template",
    )

    parser.add_argument(
        "--release",
        action="store_true",
        help="Create release tag and notes",
    )

    parser.add_argument(
        "--announce",
        action="store_true",
        help="Show announcement message template",
    )

    # Deprecated --step flag (for backwards compatibility)
    parser.add_argument(
        "--step",
        choices=[Step.PREPARE, Step.BUG, Step.TESTS, Step.SUMMARY, Step.RELEASE, Step.ANNOUNCE],
        help=argparse.SUPPRESS,  # Hide from help
    )

    args = parser.parse_args()

    # Determine version to use
    if args.version:
        # Explicit version specified - use it
        version = args.version
        print(f"Using explicit version: {version}")
    elif args.new:
        # Force new release
        version = generate_version()
        print(f"Starting new release: {version}")
    else:
        # Smart default: continue if in progress, new if last was released
        recent = get_most_recent_state()
        if recent:
            recent_version, recent_state = recent
            if recent_state.released:
                # Last release was completed, start new one
                version = generate_version()
                print(f"Previous release ({recent_version}) completed. Starting new: {version}")
            else:
                # Continue in-progress release
                version = recent_version
                print(f"Continuing in-progress release: {version}")
        else:
            # No existing state, start new
            version = generate_version()
            print(f"Starting new release: {version}")

    # Print header with version and contacts
    print("=" * 80)
    print(bold(cyan(f"Stable Release System - Version {version}")))
    print("=" * 80)
    print("\nContacts:")
    for contact in CONTACTS:
        print(f"  - {contact}")
    print("")

    # Determine which steps to run
    steps_to_run = []

    # Check for new flag-based args
    if args.prepare_branch:
        steps_to_run.append(("prepare-branch", {}))
    if args.bugs:
        steps_to_run.append(("bugs", {}))
    if args.workflow is not None:  # Can be empty string for "run all workflows"
        steps_to_run.append(("workflow", {"workflow_filter": args.workflow or None}))
    if args.summary:
        steps_to_run.append(("summary", {}))
    if args.release:
        steps_to_run.append(("release", {}))
    if args.announce:
        steps_to_run.append(("announce", {}))

    # Handle deprecated --step flag for backwards compatibility
    if args.step:
        step_map = {
            Step.PREPARE: ("prepare-branch", {}),
            Step.BUG: ("bugs", {}),
            Step.TESTS: ("workflow", {"workflow_filter": None}),
            Step.SUMMARY: ("summary", {}),
            Step.RELEASE: ("release", {}),
            Step.ANNOUNCE: ("announce", {}),
        }
        steps_to_run.append(step_map[args.step])

    # If no steps specified, run full release flow
    if not steps_to_run:
        steps_to_run = [
            ("prepare-branch", {}),
            ("bugs", {}),
            ("workflow", {"workflow_filter": None}),
            ("summary", {}),
            ("release", {}),
            ("announce", {}),
        ]

    # Map step names to functions
    step_functions = {
        "prepare-branch": step_prepare_branch,
        "bugs": step_bug_check,
        "workflow": step_workflow_tests,
        "summary": step_summary,
        "release": step_release,
        "announce": step_announce,
    }

    # Execute steps
    for step_name, step_kwargs in steps_to_run:
        func = step_functions[step_name]
        # Merge version with step-specific kwargs
        all_kwargs = {"version": version, **step_kwargs}
        func(**all_kwargs)


if __name__ == "__main__":
    main()
