#!/usr/bin/env -S uv run
"""Stable Release System (lean, single-file)

Usage:
  ./devops/stable/release.py --all                      # run full flow
  ./devops/stable/release.py --step workflow-tests      # quick mode (50k)
  ./devops/stable/release.py --step workflow-tests --comprehensive  # staging mode (2B)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Optional

from devops.job_runner import run_local, run_remote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

Location = Literal["local", "remote"]
Outcome = Literal["passed", "failed", "skipped", "inconclusive"]


class WorkflowType(StrEnum):
    """Types of workflow validations."""

    TRAIN = "train"  # Multi-GPU training validation
    TEST = "test"  # Run metta test locally
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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _get_git_log_since_stable() -> str:
    """Get git log since the last stable release tag."""
    try:
        # Find the latest stable tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match=v*", "stable"],
            check=True,
            capture_output=True,
            text=True,
        )
        last_tag = result.stdout.strip()

        # Get git log from that tag to HEAD
        result = subprocess.run(
            ["git", "log", f"{last_tag}..HEAD", "--oneline"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # If no stable tag found, just return recent commits
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
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
    """Load release state from JSON file."""
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
        )
        return state
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return None


def save_state(state: ReleaseState) -> Path:
    """Save release state to JSON file."""
    path = STATE_DIR / f"{state.version}.json"
    serialized = _recursive_asdict(state)
    path.write_text(json.dumps(serialized, indent=2))
    return path


# ============================================================================
# Metrics Extraction & Acceptance Criteria
# ============================================================================

# Regex patterns for extracting metrics from logs
_SPS_RE = re.compile(r"\bSPS[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_EVAL_RATE_RE = re.compile(r"\beval[_\s-]?success[_\s-]?rate[:=]\s*(0?\.\d+|1(?:\.0)?)", re.IGNORECASE)


def extract_metrics(log_text: str) -> dict[str, float]:
    """Extract metrics from log text using regex patterns.

    Supports:
    - SPS (samples per second) - max and last value
    - Eval success rate
    """
    metrics: dict[str, float] = {}

    # Extract SPS values
    sps_matches = [float(x) for x in _SPS_RE.findall(log_text)]
    if sps_matches:
        metrics["sps_max"] = max(sps_matches)
        metrics["sps_last"] = sps_matches[-1]

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
    # Skip if already completed
    if validation.name in state.validations:
        existing = state.validations[validation.name]
        if existing.outcome in ("passed", "failed", "skipped"):
            logger.info(f"  ⏭️  {validation.name} - already completed ({existing.outcome})")
            return existing

    # Create new result record (for state persistence)
    result = ValidationResult(
        name=validation.name,
        workflow_type=validation.workflow_type,
        location=validation.location,
        started_at=datetime.utcnow().isoformat(timespec="seconds"),
    )

    try:
        logger.info(f"  🔄 {validation.name} [{validation.workflow_type.value}] - starting...")

        # Handle different workflow types
        if validation.workflow_type == WorkflowType.TEST:
            # Run metta test locally via job_runner
            cmd = ["metta", "test"]
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
            logger.info("     Launching play window for manual verification...")
            logger.info(f"     Command: {' '.join(cmd)}")

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
                logger.info("     Play window launched successfully")
                if not _get_user_confirmation(f"Did the play workflow '{validation.name}' work correctly?"):
                    result.complete("failed", 1, error="User indicated play workflow failed")
                    logger.error(f"  ❌ {validation.name} - User confirmed FAILURE")
                    return result

        elif validation.workflow_type == WorkflowType.TRAIN:
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

                logger.info(f"     Cluster config: {nodes} nodes × {gpus} GPUs = {nodes * gpus} total GPUs")

                job = run_remote(
                    name=validation.name,
                    module=validation.module,
                    args=validation.args,
                    timeout_s=validation.timeout_s,
                    log_dir=str(LOG_DIR_REMOTE),
                    base_args=base_args,
                )
                result.exit_code = job.wait(timeout_s=validation.timeout_s, stream_output=True)
                # Extract data from RemoteJob for state persistence
                log_text = job.get_logs()
                result.logs_path = job.logs_path
                result.job_id = job.job_id

        # Check exit code
        if result.exit_code == 124:
            result.complete("failed", result.exit_code, error="Timeout exceeded")
            logger.error(f"  ❌ {validation.name} - TIMEOUT")
            return result
        elif result.exit_code != 0:
            result.complete("failed", result.exit_code, error=f"Exit code {result.exit_code}")
            logger.error(f"  ❌ {validation.name} - FAILED (exit {result.exit_code})")
            return result

        # Extract metrics (only for TRAIN workflows)
        if validation.workflow_type == WorkflowType.TRAIN:
            result.metrics = extract_metrics(log_text or "")

        # Evaluate acceptance criteria
        if validation.acceptance:
            outcome, failed_checks = evaluate_thresholds(result.metrics, validation.acceptance)
            result.complete(outcome, result.exit_code)

            if outcome == "passed":
                logger.info(f"  ✅ {validation.name} - PASSED")
                if result.metrics:
                    metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
                    logger.info(f"     Metrics: {metrics_str}")
            else:
                logger.error(f"  ❌ {validation.name} - FAILED acceptance criteria")
                if result.metrics:
                    metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
                    logger.info(f"     Metrics: {metrics_str}")
                for check in failed_checks:
                    logger.error(f"     Failed: {check.note}")
        else:
            # No acceptance criteria - just mark as passed
            result.complete("passed", result.exit_code)
            logger.info(f"  ✅ {validation.name} - PASSED")

        return result

    except subprocess.TimeoutExpired as e:
        result.complete("failed", 124, error=f"Timeout: {e}")
        logger.error(f"  ❌ {validation.name} - TIMEOUT")
        return result
    except Exception as e:
        result.complete("failed", 1, error=f"Unexpected error: {e}")
        logger.error(f"  ❌ {validation.name} - ERROR: {e}")
        return result


# ============================================================================
# Release Plans
# ============================================================================


def get_workflow_tests() -> list[Validation]:
    """Get the validation plan for this release.

    Returns:
        List of validations to run (TEST, TRAIN single-GPU, TRAIN multi-GPU, PLAY)
    """
    validations = []

    # 1. TEST workflow - run metta test locally
    validations.append(
        Validation(
            name="metta_test",
            workflow_type=WorkflowType.TEST,
            module="",  # Not used for TEST workflow
            location="local",
            timeout_s=1800,  # 30 minutes
        )
    )

    # 2. TRAIN workflow - local smoke test
    validations.append(
        Validation(
            name="arena_local_smoke",
            workflow_type=WorkflowType.TRAIN,
            module="experiments.recipes.arena_basic_easy_shaped.train",
            location="local",
            args=["run=stable.smoke", "trainer.total_timesteps=1000", "wandb.enabled=false"],
            timeout_s=600,
            acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=30000)],
        )
    )

    # 3. TRAIN workflow - single GPU remote validation
    validations.append(
        Validation(
            name="arena_single_gpu_50k",
            workflow_type=WorkflowType.TRAIN,
            module="experiments.recipes.arena_basic_easy_shaped.train",
            location="remote",
            args=["trainer.total_timesteps=50000"],
            timeout_s=3600,  # 1 hour
            acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=40000)],
        )
    )

    # 4. TRAIN workflow - multi-GPU remote validation
    validations.append(
        Validation(
            name="arena_multi_gpu_2b",
            workflow_type=WorkflowType.TRAIN,
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
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        logger.error("Not in a git repository. Aborting.")
        sys.exit(1)


def step_prepare_branch(version: str, **_kwargs) -> None:
    """Step 1: Create and push staging branch."""
    _ensure_git_repo()

    branch_name = f"staging/v{version}-rc1"

    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 1: Prepare Staging Branch (v{version})")
    logger.info("=" * 60 + "\n")

    logger.info(f"Creating branch: {branch_name}")
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True, capture_output=True)
        logger.info(f"✅ Branch {branch_name} created and pushed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create/push branch: {e}")
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
        logger.warning("Asana library not installed (pip install asana)")
        return None

    try:
        # Initialize Asana client
        config = asana.Configuration()
        config.access_token = token
        client = asana.ApiClient(config)

        # Get user info to verify auth
        users_api = asana.UsersApi(client)
        user = users_api.get_user("me", {})
        logger.info(f"✓ Authenticated as {user.get('name', '?')}")

        # Get project sections
        sections_api = asana.SectionsApi(client)
        sections = sections_api.get_sections_for_project(project_id, {})

        # Find "Active" section
        active_section = next((s for s in sections if s["name"].lower() == "active"), None)
        if not active_section:
            logger.warning("No 'Active' section found in Asana project")
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
            logger.info("✅ No blocking bugs in Asana Active section")
            return True

        logger.error(f"❌ Found {len(open_tasks)} blocking task(s) in Active:")
        for task in open_tasks[:10]:  # Show first 10
            logger.error(f"  • {task['name']}")
            if task.get("permalink_url"):
                logger.error(f"    {task['permalink_url']}")
        return False

    except Exception as e:
        logger.warning(f"Asana API error: {e}")
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
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Bug Status Check")
    logger.info("=" * 60 + "\n")

    # Try automated Asana check
    result = _check_asana_blockers()

    if result is True:
        logger.info("✅ Bug check PASSED - clear for release")
        return
    elif result is False:
        logger.error("❌ Bug check FAILED - resolve blocking issues before release")
        sys.exit(1)

    # Asana check inconclusive - fall back to manual
    logger.warning("⚠️  Asana automation unavailable or inconclusive")
    logger.info("\nTo enable automated checking:")
    logger.info("  export ASANA_TOKEN='your_personal_access_token'")
    logger.info("  export ASANA_PROJECT_ID='your_project_id'")
    logger.info("\nManual steps required:")
    logger.info("1. Open Asana project for bug tracking")
    logger.info("2. Verify no active/open bugs marked as blockers")
    logger.info("3. Update bug statuses as needed in consultation with bug owners")
    logger.info("")

    if not _get_user_confirmation("Have you completed the bug status check and is it PASSED?"):
        logger.error("❌ Bug check FAILED - user indicated issues remain")
        sys.exit(1)

    logger.info("✅ Bug check PASSED - user confirmed")


def step_workflow_tests(version: str, **_kwargs) -> None:
    """Step 3: Run validation workflows."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Workflow Validation")
    logger.info("=" * 60 + "\n")

    # Load or create state
    state_version = f"release_{version}"
    state = load_state(state_version)
    if not state:
        state = ReleaseState(
            version=state_version,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            commit_sha=_get_commit_sha(),
        )

    # Run validations sequentially
    validations = get_workflow_tests()
    for validation in validations:
        # Determine cluster config based on validation name
        if "multi_gpu" in validation.name:
            cluster_config = {"nodes": 4, "gpus": 4}
        elif "single_gpu" in validation.name:
            cluster_config = {"nodes": 1, "gpus": 1}
        else:
            cluster_config = None

        result = run_validation(state, validation, cluster_config=cluster_config)
        state.validations[validation.name] = result
        save_state(state)

    # Print summary
    passed = sum(1 for r in state.validations.values() if r.outcome == "passed")
    failed = sum(1 for r in state.validations.values() if r.outcome == "failed")

    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    logger.info("\nResults:")
    logger.info(f"  ✅ Passed:  {passed}")
    logger.info(f"  ❌ Failed:  {failed}")

    logger.info("\nDetailed Results:")
    for result in state.validations.values():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(result.outcome or "", "❓")
        logger.info(f"  {icon} {result.name:24} loc={result.location:6} exit={result.exit_code:>3}")
        if result.metrics:
            metrics_str = "  ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            logger.info(f"       Metrics: {metrics_str}")
        if result.logs_path:
            logger.info(f"       Logs: {result.logs_path}")
        if result.job_id:
            logger.info(f"       Job ID: {result.job_id}")

    logger.info("=" * 80)

    if failed:
        logger.info(f"\nState saved to: {STATE_DIR / f'{state_version}.json'}")
        logger.error("❌ Workflow validation FAILED")
        sys.exit(1)

    logger.info("\n✅ All workflow validations PASSED")
    logger.info(f"State saved to: {STATE_DIR / f'{state_version}.json'}")


def step_release(version: str, **_kwargs) -> None:
    """Step 4: Print release instructions with actual metrics from validation."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Release")
    logger.info("=" * 60 + "\n")

    # Load state to extract metrics
    state_version = f"release_{version}"
    state = load_state(state_version)

    # Extract training metrics from TRAIN validations
    training_metrics = {}
    training_job_id = None
    for name, result in state.validations.items() if state else {}:
        if "train" in name.lower() and result.metrics:
            training_metrics.update(result.metrics)
            if result.job_id:
                training_job_id = result.job_id

    # Format metrics for display
    sps_max = training_metrics.get("sps_max", "N/A")
    sps_last = training_metrics.get("sps_last", "N/A")

    # Get git log since last stable release
    git_log = _get_git_log_since_stable()

    logger.info("4.1 Prepare Release PR:")
    logger.info(f"  1. Create release notes at: devops/stable/release-notes/v{version}.md")
    logger.info(f"  2. Open PR from staging/v{version}-rc1 to stable")
    logger.info("  3. Use this template for PR description:")
    logger.info("")
    logger.info(f"## Version {version}")
    logger.info("")
    logger.info("### Changes Since Last Stable Release")
    logger.info("")
    if git_log:
        for line in git_log.split("\n"):
            logger.info(f"- {line}")
    else:
        logger.info("- <No commits found>")
    logger.info("")
    logger.info("### Known Issues")
    logger.info("")
    logger.info("<Notes from step 2>")
    logger.info("")
    logger.info("### Training Job Links")
    logger.info("")
    if training_job_id:
        logger.info(f"- SkyPilot Job ID: {training_job_id}")
        logger.info(f"- View logs: sky logs {training_job_id}")
    else:
        logger.info("- Job ID: <not available>")
    logger.info("")
    logger.info("### Key Metrics")
    logger.info("")
    logger.info(f"- Training throughput (SPS max): {sps_max}")
    logger.info(f"- Training throughput (SPS last): {sps_last}")
    logger.info("- Final loss: <manual - check logs>")
    logger.info("- Final reward: <manual - check logs>")
    logger.info("- Elapsed training time: <manual - check logs>")
    logger.info("- GPU usage: <manual - check logs>")
    logger.info("- CPU usage: <manual - check logs>")
    logger.info("- GPU memory usage: <manual - check logs>")
    logger.info("- CPU memory usage: <manual - check logs>")
    logger.info("")
    logger.info("4.2 Merge and Tag:")
    logger.info("  1. After PR approval, merge into stable")
    logger.info("  2. Checkout stable and create the final annotated release tag:")
    logger.info("")
    logger.info("     git checkout stable")
    logger.info("     git pull")
    logger.info(f'     git tag -a v{version} -m "Release version {version}"')
    logger.info(f"     git push origin v{version}")


def step_announce(version: str, **_kwargs) -> None:
    """Step 5: Print announcement instructions."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Announce")
    logger.info("=" * 60 + "\n")

    logger.info("Post release completion to Discord in #eng-process")
    logger.info("\nSuggested message:")
    logger.info(f"Released stable version v{version}")
    logger.info(f"Release notes: devops/stable/release-notes/v{version}.md")


# ============================================================================
# CLI
# ============================================================================


class Step:
    """Step identifiers for the release process."""

    PREPARE = "prepare-branch"
    BUG = "bug-check"
    TESTS = "workflow-tests"
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
    )

    parser.add_argument(
        "--version",
        help="Version number (default: auto-generated from date YYYY.MM.DD-HHMM)",
        default=None,
    )

    parser.add_argument(
        "--step",
        choices=[Step.PREPARE, Step.BUG, Step.TESTS, Step.RELEASE, Step.ANNOUNCE],
        help="Run a specific step",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps",
    )

    args = parser.parse_args()

    # Generate or use provided version
    version = args.version if args.version else generate_version()

    # Show help if no action specified
    if not args.step and not args.all:
        parser.print_help()
        logger.info(f"\nAuto-generated version: {version}")
        logger.info("\nEnvironment variables:")
        logger.info("  ASANA_TOKEN - Personal Access Token for Asana API")
        logger.info("  ASANA_PROJECT_ID - Asana project ID for bug tracking")
        logger.info("\nValidations:")
        logger.info("  - TEST: metta test locally")
        logger.info("  - TRAIN: local smoke, single-GPU remote, multi-GPU remote")
        logger.info("  - PLAY: interactive testing with manual confirmation")
        logger.info("\nContacts:")
        for contact in CONTACTS:
            logger.info(f"  - {contact}")
        return

    # Map steps to functions (all have same signature now)
    steps = {
        Step.PREPARE: step_prepare_branch,
        Step.BUG: step_bug_check,
        Step.TESTS: step_workflow_tests,
        Step.RELEASE: step_release,
        Step.ANNOUNCE: step_announce,
    }

    # Build kwargs (same for all steps)
    kwargs = {
        "version": version,
    }

    # Print header with version and contacts
    logger.info("=" * 80)
    logger.info(f"Stable Release System - Version {version}")
    logger.info("=" * 80)
    logger.info("\nContacts:")
    for contact in CONTACTS:
        logger.info(f"  - {contact}")
    logger.info("")

    # Execute
    if args.all:
        for step_name in [Step.PREPARE, Step.BUG, Step.TESTS, Step.RELEASE, Step.ANNOUNCE]:
            steps[step_name](**kwargs)
    else:
        steps[args.step](**kwargs)


if __name__ == "__main__":
    main()
