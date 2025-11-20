#!/usr/bin/env python3
"""Stable Release System - CLI

Commands:
  validate              Run validation only (prepare-tag -> validation -> summary)
  release               Full release pipeline (prepare-tag -> validation -> bug check -> release tag)
  hotfix                Hotfix mode (prepare-tag -> release, skip validation)

Options:
  --version X           Use specific version
  --new                 Force new release (ignore in-progress state)
  --skip-commit-match   Skip verification that current commit matches RC tag
  --job PATTERN        Filter validation jobs (validate and release modes)
  --retry-failed        Retry failed jobs (validate and release modes)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

import gitta as git
from devops.stable.asana_bugs import check_blockers
from devops.stable.display import format_job_result, format_training_job_section
from devops.stable.jobs import get_all_jobs
from devops.stable.state import (
    Gate,
    ReleaseState,
    get_most_recent_state,
    load_or_create_state,
    load_state,
    save_state,
)
from metta.common.util.fs import get_repo_root
from metta.common.util.text_styles import bold, cyan, green, yellow
from metta.jobs.job_api import monitor_jobs_until_complete, submit_monitor_and_report
from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import ExitCode, JobManager
from metta.jobs.job_state import JobStatus

# ============================================================================
# Constants
# ============================================================================

CONTACTS = [
    "Release Manager: @Robb",
    "Technical Lead: @Jack Heart",
    "Bug Triage: @Nishad Singh",
]

# ============================================================================
# Helper Functions
# ============================================================================


def print_step_header(title: str, width: int = 60) -> None:
    """Print formatted step header."""
    print("\n" + "=" * width)
    print(bold(title))
    print("=" * width + "\n")


def is_step_complete(state: ReleaseState | None, step_name: str) -> bool:
    """Check if a pipeline step is already completed in state."""
    if not state:
        return False
    return any(g.step == step_name and g.passed for g in state.gates)


def mark_step_complete(state: ReleaseState | None, step_name: str) -> None:
    """Mark a pipeline step as completed in state."""
    if not state:
        return
    state.gates.append(
        Gate(
            step=step_name,
            passed=True,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
    )
    save_state(state)


def get_user_confirmation(prompt: str, default: Optional[bool] = None, no_interactive: bool = False) -> bool:
    """Get yes/no confirmation from user.

    Args:
        prompt: Question to ask user
        default: Default value if running non-interactively. If None, will fail in non-interactive mode.
        no_interactive: If True, use default without prompting

    Returns:
        True if user confirms, False otherwise

    Raises:
        RuntimeError: If default is None and running in non-interactive mode
    """
    # Check if we're in non-interactive mode
    if no_interactive:
        if default is None:
            raise RuntimeError(f"Cannot prompt '{prompt}' in non-interactive mode without a default")
        return default

    # Build prompt suffix based on default
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    while True:
        try:
            response = input(f"{prompt}{suffix}").strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            elif response == "" and default is not None:
                return default
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except (EOFError, KeyboardInterrupt):
            return default if default is not None else False


def setup_logging(log_file: Path) -> None:
    """Configure logging to write to file.

    All log messages (including from background threads) will be written to the log file.
    User can tail the log file in another terminal to see detailed progress.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler for all logs
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    # Configure root logger: remove all handlers and add only file handler
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]  # Replace all handlers (removes console output)

    # Set metta logger to DEBUG (captures all metta.* logs in detail)
    # Other loggers will use their default levels (typically WARNING)
    metta_logger = logging.getLogger("metta")
    metta_logger.setLevel(logging.DEBUG)


def state_dir() -> Path:
    return get_repo_root() / "devops/stable/state"


def version_state_dir(version: str) -> Path:
    """Get version-specific state directory."""
    return state_dir() / version


def log_file() -> Path:
    """Get unified log file path (shared across all versions)."""
    return state_dir() / "job_manager.log"


def get_job_manager(version: str) -> JobManager:
    """Get JobManager instance for release validation (version-specific database, unified logs)."""
    return JobManager(base_dir=version_state_dir(version))


def load_state_or_exit(version: str, step_name: str) -> ReleaseState:
    """Load release state or exit with error message."""
    state_version = f"v{version}"
    state = load_state(state_version)
    assert state, f"State should have been created before '{step_name}'"
    return state


def get_rc_commit(version: str) -> str:
    """Get the commit SHA that the RC tag points to.

    Args:
        version: Release version (without 'v' prefix)

    Returns:
        Commit SHA that the RC tag points to

    Raises:
        AssertionError: If RC tag doesn't exist (indicates bug in control flow)
    """
    rc_tag_name = f"v{version}-rc"
    try:
        return git.run_git("rev-list", "-n", "1", rc_tag_name).strip()
    except git.GitError as e:
        raise AssertionError(f"RC tag {rc_tag_name} should have been created by prepare_tag step") from e


def verify_on_rc_commit(version: str, step_name: str, skip_check: bool = False) -> str:
    """Verify we're on the commit that the RC tag points to.

    Args:
        version: Release version
        step_name: Name of current step (for error messages)
        skip_check: If True, skip the commit match verification (show warning instead)

    Returns:
        RC commit SHA

    Raises:
        SystemExit: If not on RC commit or RC tag doesn't exist
    """
    rc_tag_name = f"v{version}-rc"

    # Get current commit
    current_commit = git.get_current_commit()

    # Get RC tag commit
    rc_commit = get_rc_commit(version)

    # Verify we're on the RC commit (unless skip_check is set)
    if current_commit != rc_commit:
        if skip_check:
            print(yellow(f"⚠️  WARNING: Not on RC commit for {step_name} (check skipped with --skip-commit-match)"))
            print(f"   Current commit:  {current_commit}")
            print(f"   RC tag commit:   {rc_commit} ({rc_tag_name})")
        else:
            print(f"❌ ERROR: Not on RC commit for {step_name}")
            print(f"   Current commit:  {current_commit}")
            print(f"   RC tag commit:   {rc_commit} ({rc_tag_name})")
            print("\n   To fix, checkout the RC commit:")
            print(f"   git checkout {rc_tag_name}")
            print("   Or use --skip-commit-match to skip this check")
            sys.exit(1)

    return rc_commit


# ============================================================================
# Release Pipeline Steps
# ============================================================================
# State management architecture:
#
# ReleaseState (state/{version}.json):
#   - Tracks pipeline-level gates: prepare_tag, bug_check
#   - Allows pipeline-level resume (which high-level steps to skip)
#   - Lightweight: just step names, timestamps, pass/fail
#
# JobManager (jobs.sqlite):
#   - Tracks individual validation job execution
#   - Job states: pending, running, completed
#   - Exit codes: 0=success, >0=error, -1=abnormal, 130=cancelled
#   - Handles concurrency, logs, metrics extraction
#   - Restart behavior:
#     * Local jobs: Stale PIDs detected and auto-retried
#     * Remote jobs: Reattach to running jobs on cluster
#
# RC tag workflow:
#   - prepare_tag: Creates v{version}-rc tag to mark validation commit
#   - All validation steps: Verify on RC commit (ensures consistency)
#   - release: Creates final v{version} tag pointing to same commit
#
# UX flow:
#   - JobDisplay displays live status during job execution
#   - Monitor updates in-place (preserves step headers)
#   - Final summary shows all artifacts (WandB, checkpoints)
#   - Script only prints simple verdict after monitor summary
# ============================================================================


def _prepare_jobs_for_release(
    job_configs: list[JobConfig],
    state_version: str,
    job_manager: JobManager,
    retry: bool,
) -> list[JobConfig]:
    """Prepare job configs for release validation with version prefixing and retry logic.

    Returns list of prepared JobConfig objects ready to submit.
    """
    prepared_jobs = []

    for job_config in job_configs:
        # Job names are already fully qualified by get_stable_jobs()
        job_name = job_config.name

        # Check if job already exists
        existing_state = job_manager.get_job_state(job_name)

        if existing_state and existing_state.status == "completed":
            # Determine if we should retry
            should_retry = False
            if existing_state.exit_code in (-1, 130):
                # Abnormal termination (-1) or interrupted (130/SIGINT) - always retry
                reason = "abnormal termination" if existing_state.exit_code == -1 else "interrupted (SIGINT)"
                print(yellow(f"🔄 {job_config.name} - retrying after {reason}"))
                should_retry = True
            elif retry and (existing_state.exit_code != 0 or not existing_state.is_successful):
                # Failed and retry requested
                print(yellow(f"🔄 {job_config.name} - retrying previous run"))
                should_retry = True
            else:
                # Already completed successfully or retry not requested
                print(f"⏭️  {job_config.name} - already completed (use --retry to retry)")
                continue

            if should_retry:
                job_manager.delete_job(job_name)
        elif existing_state and existing_state.status in ("pending", "running"):
            print(f"⏭️  {job_config.name} - already {existing_state.status}, will reattach")
            # Don't add to prepared_jobs since it's already submitted
            continue

        # Set group for monitoring
        # Note: Job names and args are already fully qualified by get_stable_jobs()
        job_config.group = state_version

        prepared_jobs.append(job_config)

    return prepared_jobs


def step_prepare_tag(
    version: str, state: Optional[ReleaseState] = None, no_interactive: bool = False, **_kwargs
) -> None:
    """Create staging tag (v{version}-rc) to mark commit for validation."""
    tag_name = f"v{version}-rc"
    print_step_header(f"Prepare Staging Tag (v{version})")

    # Skip if already completed
    if is_step_complete(state, "prepare_tag"):
        print(green(f"✅ Step already completed - tag {tag_name} exists"))
        return

    commit_sha = git.get_current_commit()
    print(f"Current commit: {commit_sha}\n")

    # Check if tag already exists
    existing = git.run_git("tag", "-l", tag_name).strip()
    if existing:
        print(f"⚠️  Tag {tag_name} already exists")
        if state:
            # Continuation from previous run - mark complete
            print(green("   Marking as complete from previous run"))
            mark_step_complete(state, "prepare_tag")
            return
        # Fresh run - ask user if we should recreate
        # Default to No - safer to not delete existing tags automatically
        if not get_user_confirmation("Delete existing tag and continue?", default=False, no_interactive=no_interactive):
            sys.exit(1)
        git.run_git("tag", "-d", tag_name)
        try:
            git.run_git("push", "origin", f":refs/tags/{tag_name}")
        except git.GitError:
            pass  # Remote tag didn't exist

    # Create and push tag
    print(f"Creating staging tag: {tag_name}")
    git.run_git("tag", tag_name)
    git.run_git("push", "origin", tag_name)
    print(green(f"✅ Tag {tag_name} created and pushed\n"))

    mark_step_complete(state, "prepare_tag")


def step_bug_check(
    version: str,
    state: Optional[ReleaseState] = None,
    skip_commit_match: bool = False,
    no_interactive: bool = False,
    **_kwargs,
) -> None:
    """Check for blocking bugs via Asana or manual confirmation."""
    print_step_header("Bug Status Check")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "bug check", skip_check=skip_commit_match)
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Skip if already completed
    if is_step_complete(state, "bug_check"):
        print(green("✅ Step already completed"))
        return

    # Try automated Asana check
    result = check_blockers()

    if result is True:
        print("✅ Bug check PASSED - clear for release")
        mark_step_complete(state, "bug_check")
        return
    elif result is False:
        print("❌ Bug check FAILED - resolve blocking issues before release")
        sys.exit(1)

    # Asana check inconclusive - fall back to manual or skip in non-interactive mode
    if no_interactive:
        # In non-interactive mode, log that we're skipping Asana check and proceed
        print("⚠️  Asana automation unavailable - skipping bug check in non-interactive mode")
        print("✅ Bug check SKIPPED (non-interactive mode)")
        mark_step_complete(state, "bug_check")
        return

    # Interactive mode - prompt user
    print("⚠️  Asana automation unavailable")
    print("\nManual verification required:")
    print("  1. Check Asana for blocking bugs")
    print("  2. Resolve or triage any blockers\n")

    if not get_user_confirmation("Bug check PASSED?", default=True, no_interactive=False):
        print("❌ Bug check FAILED")
        sys.exit(1)

    print("✅ Bug check PASSED")
    mark_step_complete(state, "bug_check")


def step_job_validation(
    version: str,
    job: Optional[str] = None,
    retry: bool = False,
    skip_commit_match: bool = False,
    no_interactive: bool = False,
    **_kwargs,
) -> None:
    """Run validation jobs via JobManager.

    JobManager handles execution, state persistence, metrics extraction, and dependency resolution.
    Monitor displays live status with in-place updates.
    """
    print_step_header("Task Validation")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "job validation", skip_check=skip_commit_match)
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and filter jobs
    state_version = f"v{version}"
    load_or_create_state(state_version, git.get_current_commit())
    all_job_configs = get_all_jobs(version=state_version)

    if job:
        job_configs = [t for t in all_job_configs if job in t.name]
        if not job_configs:
            print(f"❌ No jobs matching '{job}'")
            print(f"Available: {', '.join(t.name for t in all_job_configs)}")
            sys.exit(1)
        print(f"Running: {len(job_configs)} job(s) matching '{job}'\n")
    else:
        job_configs = all_job_configs
        print(f"Running: {len(job_configs)} job(s)\n")

    # Initialize JobManager
    job_manager = get_job_manager(version)
    log_path = log_file()
    print(f"💡 Detailed logs: tail -f {log_path}\n")

    # Prepare jobs with version prefixing and retry logic
    prepared_jobs = _prepare_jobs_for_release(job_configs, state_version, job_manager, retry)

    # Submit new jobs if any
    if prepared_jobs:
        submit_monitor_and_report(
            job_manager,
            prepared_jobs,
            title=f"Release Validation: {state_version}",
            group=state_version,
            no_interactive=no_interactive,
        )
    else:
        print("No new jobs to submit")

    # Now wait for ALL jobs in the group (including already-running ones)
    print("\nWaiting for all jobs to complete...")
    all_job_names = [job_config.name for job_config in job_configs]
    monitor_jobs_until_complete(
        all_job_names,
        job_manager,
        title=f"Release Validation: {state_version}",
        group=state_version,
        no_interactive=no_interactive,
    )

    # Show detailed release-specific displays (acceptance + training artifacts)
    print("\n" + "=" * 80)
    print("Detailed Job Information")
    print("=" * 80 + "\n")

    job_config_by_name = {config.name: config for config in job_configs}
    jobs_dict = job_manager.get_group_jobs(state_version)

    for job_state in jobs_dict.values():
        # Extract job name from job name (format: {version}_{job_name})
        job_name = job_state.name.split("_", 1)[1] if "_" in job_state.name else job_state.name
        job_config = job_config_by_name.get(job_name)

        if not job_config:
            continue

        # Show acceptance criteria and training artifacts
        display = format_job_result(job_state)
        print(display)
        print()

    # Count results (distinguish failed vs running vs skipped)
    passed = 0
    failed = 0
    running = 0
    skipped = 0

    for job_config in job_configs:
        job_state = job_manager.get_job_state(job_config.name)
        if not job_state:
            skipped += 1
        elif job_state.exit_code == ExitCode.SKIPPED:
            skipped += 1
        elif job_state.status == JobStatus.RUNNING:
            running += 1
        elif job_state.is_successful:
            passed += 1
        else:
            failed += 1

    # Print verdict
    print()

    # Check if jobs are still running
    if running > 0:
        msg = f"⏳ Task validation IN PROGRESS ({passed} passed, {running} still running"
        if failed > 0:
            msg += f", {failed} failed"
        if skipped > 0:
            msg += f", {skipped} skipped"
        msg += ")"
        print(msg)
        print("\n⚠️  Cannot proceed to summary - jobs still running")
        print("   Re-run this command to check status, or wait for jobs to complete")
        sys.exit(1)

    if failed > 0:
        msg = f"❌ Task validation FAILED ({passed} passed, {failed} failed"
        if skipped > 0:
            msg += f", {skipped} skipped"
        msg += ")"
        print(msg)
        sys.exit(1)

    if skipped > 0:
        print(f"⚠️  Task validation incomplete ({passed} passed, {skipped} skipped)")
        sys.exit(1)

    print(f"✅ All job validations PASSED ({passed}/{len(job_configs)})")


def step_summary(version: str, skip_commit_match: bool = False, **_kwargs) -> None:
    """Print validation summary and release notes template.

    Queries JobManager for job outcomes and metrics.
    """
    print_step_header("Release Summary")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "summary", skip_check=skip_commit_match)
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state version and JobManager
    state_version = f"v{version}"
    _ = load_state_or_exit(version, "summary")  # Verify state exists
    job_manager = get_job_manager(version)

    # Get git log since last stable release
    git_log = git.git_log_since("origin/stable")

    # Get all jobs to display results and collect training job info
    all_job_configs = get_all_jobs(version=state_version)
    training_jobs = []  # Track training jobs separately

    # Print job results summary
    print("Task Results:")
    for job_config in all_job_configs:
        job_name = job_config.name  # Already fully qualified
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            # Determine icon based on status
            if job_state.status == JobStatus.RUNNING:
                icon = "⏳"
            elif job_state.is_successful:
                icon = "✅"
            else:
                icon = "❌"
            print(f"  {icon} {job_config.name}")
            if job_state.metrics:
                # Filter out internal metrics (starting with _) and format numeric values
                metrics_str = ", ".join(
                    f"{k}={v:.1f}"
                    for k, v in job_state.metrics.items()
                    if not k.startswith("_") and isinstance(v, (int, float))
                )
                if metrics_str:
                    print(f"       Metrics: {metrics_str}")

            # Collect training job info
            if job_config.is_training_job:
                training_jobs.append((job_config.name, job_state))

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
    print("### Training Jobs")
    print("")
    if training_jobs:
        for job_name, job_state in training_jobs:
            formatted = format_training_job_section(job_name, job_state)
            print(formatted)
            print("")
    else:
        print("- No training jobs in this validation run")
    print("")


def step_release(version: str, skip_commit_match: bool = False, **_kwargs) -> None:
    """Create release tag and release notes.

    Verifies all validation jobs passed before creating release.
    """
    print_step_header("Create Release")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "release", skip_check=skip_commit_match)
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and JobManager
    state_version = f"v{version}"
    state = load_state_or_exit(version, "release")
    job_manager = get_job_manager(version)

    # Verify all jobs passed
    all_job_configs = get_all_jobs(version=state_version)

    failed_jobs = []
    for job_config in all_job_configs:
        job_name = job_config.name  # Already fully qualified
        job_state = job_manager.get_job_state(job_name)
        if not job_state:
            failed_jobs.append(f"{job_config.name} (not run)")
        elif not job_state.is_successful:
            failed_jobs.append(job_config.name)

    if failed_jobs:
        print("❌ Cannot release with failed/incomplete jobs:")
        for name in failed_jobs:
            print(f"  ❌ {name}")
        sys.exit(1)

    # Extract git log and training job info for release notes
    git_log = git.git_log_since("origin/stable")
    training_jobs = []

    # Create release notes
    release_notes_dir = Path("devops/stable/release-notes")
    release_notes_dir.mkdir(parents=True, exist_ok=True)
    release_notes_path = release_notes_dir / f"v{version}.md"

    release_notes_content = f"""# Release Notes - Version {version}

## Task Results Summary

"""
    # Use job results already validated above
    for job_config in all_job_configs:
        job_name = f"{state_version}_{job_config.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            if job_state.is_successful:
                icon = "✅"
            else:
                icon = "❌"
            release_notes_content += f"- {icon} {job_config.name}\n"
            if job_state.metrics:
                metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
                release_notes_content += f"  - Metrics: {metrics_str}\n"

            # Collect training jobs
            if job_config.is_training_job:
                training_jobs.append((job_config.name, job_state))

    release_notes_content += f"""
## Changes Since Last Stable Release

{git_log if git_log else "No commits found"}

## Training Jobs

"""
    if training_jobs:
        for job_name, job_state in training_jobs:
            formatted = format_training_job_section(job_name, job_state)
            release_notes_content += formatted + "\n\n"
    else:
        release_notes_content += "- No training jobs in this validation run\n"

    # Write release notes
    release_notes_path.write_text(release_notes_content)
    print(f"✅ Created release notes: {release_notes_path}")

    # Get the commit SHA from the RC tag (validation was run against this)
    rc_commit = get_rc_commit(version)
    rc_tag_name = f"v{version}-rc"
    print(f"RC tag {rc_tag_name} points to commit: {rc_commit}")

    # Create git tag pointing to the same commit as RC tag
    tag_name = f"v{version}"

    try:
        # Check if tag already exists
        existing = git.run_git("tag", "-l", tag_name)
        if existing.strip():
            print(f"Tag {tag_name} already exists")
            sys.exit(1)

        # Create annotated tag at the RC commit
        git.run_git("tag", "-a", tag_name, rc_commit, "-m", f"Release version {version}")
        print(f"✅ Created git tag: {tag_name} at {rc_commit}")

        # Push tag
        git.run_git("push", "origin", tag_name)
        print(f"✅ Pushed tag to origin: {tag_name}")

    except git.GitError as e:
        print(f"Failed to create/push tag: {e}")
        sys.exit(1)

    # Update stable branch to point to this tag
    print("\nUpdating stable branch...")
    try:
        # Fetch latest to ensure we have all tags
        git.run_git("fetch", "origin")

        # Update local stable branch to the new tag
        git.run_git("branch", "-f", "stable", tag_name)

        # Push stable branch to origin (force update)
        git.run_git("push", "origin", "stable", "--force-with-lease")

        print(f"✅ Updated origin/stable branch to {tag_name}")
    except git.GitError as e:
        print(f"⚠️  Failed to update stable branch: {e}")
        print("   You may need to update it manually:")
        print(f"   git branch -f stable {tag_name}")
        print("   git push origin stable --force-with-lease")

    # Mark state as released
    state.released = True
    save_state(state)

    # Create GitHub release using gh CLI
    print("\nCreating GitHub release...")
    try:
        # Check if gh CLI is available
        subprocess.run(["gh", "--version"], capture_output=True, check=True)

        # Create GitHub release from tag with release notes
        result = subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag_name,
                "--title",
                f"Release {version}",
                "--notes-file",
                str(release_notes_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ Created GitHub release: {tag_name}")
        print(f"   {result.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️  GitHub CLI (gh) not installed - skipping GitHub release creation")
        print("   Install: https://cli.github.com/")
        print("   Or create release manually from tag")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to create GitHub release: {e.stderr}")
        print("   You may need to authenticate: gh auth login")
        print("   Or create release manually from tag")

    print("\n" + "=" * 60)
    print("Release Complete!")
    print("=" * 60)
    print(f"\nRelease notes: {release_notes_path}")
    print(f"Git tag: {tag_name}")
    print(f"Stable branch: origin/stable -> {tag_name}")


# ============================================================================
# CLI Setup
# ============================================================================


def generate_version() -> str:
    """Generate version string from current date/time.

    Format: YYYY.MM.DD-HHMMSS (e.g., 2025.10.07-143045)

    Includes seconds to prevent version collisions when running multiple times per minute.
    """
    return datetime.now().strftime("%Y.%m.%d-%H%M%S")


def resolve_version(explicit: Optional[str], force_new: bool) -> str:
    """Determine version based on args and state.

    Returns version string (bare version without any prefix).
    State files are named "v{version}.json".
    """
    if explicit:
        print(f"Using explicit version: {explicit}")
        return explicit
    if force_new:
        v = generate_version()
        print(f"Starting new release: {v}")
        return v
    recent = get_most_recent_state()
    if recent:
        recent_version, recent_state = recent
        # Strip 'v' prefix from state filename to get bare version
        bare_version = recent_version.removeprefix("v")
        if recent_state.released:
            v = generate_version()
            print(f"Previous release ({bare_version}) completed. Starting new: {v}")
            return v
        print(f"Continuing in-progress release: {bare_version}")
        return bare_version
    v = generate_version()
    print(f"Starting new release: {v}")
    return v


app = typer.Typer(add_completion=False)


@app.callback()
def common(
    ctx: typer.Context,
    version: Optional[str] = typer.Option(None, "--version", help="Version number (overrides auto-continue behavior)"),
    new: bool = typer.Option(False, "--new", help="Force start a new release (ignore in-progress state)"),
    skip_commit_match: bool = typer.Option(
        False, "--skip-commit-match", help="Skip verification that current commit matches RC tag"
    ),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Run in non-interactive mode (use defaults for prompts)"
    ),
):
    """Stable Release System - automated release validation and deployment."""
    resolved_version = resolve_version(version, new)

    # Store in context for commands to access
    ctx.obj = {
        "version": resolved_version,
        "skip_commit_match": skip_commit_match,
        "no_interactive": no_interactive,
    }

    print("=" * 80)
    print(bold(cyan(f"Stable Release System - Version {resolved_version}")))
    print("=" * 80)
    print("\nContacts:")
    for contact in CONTACTS:
        print(f"  - {contact}")
    print("")


@app.command("validate")
def cmd_validate(
    ctx: typer.Context,
    job: Optional[str] = typer.Option(
        None,
        "--job",
        help="Filter validation jobs by name",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry previously failed jobs (default: skip failed jobs)",
    ),
):
    """Run validation pipeline (prepare-tag -> validation -> summary)."""
    version = ctx.obj["version"]
    skip_commit_match = ctx.obj["skip_commit_match"]
    no_interactive = ctx.obj["no_interactive"]

    state_version = f"v{version}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Step 1: Prepare RC tag (automatic - skips if already done)
    step_prepare_tag(version=version, state=state, no_interactive=no_interactive)

    # Step 2: Run validation
    step_job_validation(
        version=version, job=job, retry=retry, skip_commit_match=skip_commit_match, no_interactive=no_interactive
    )

    # Step 3: Show summary
    step_summary(version=version, skip_commit_match=skip_commit_match)


@app.command("hotfix")
def cmd_hotfix(ctx: typer.Context):
    """Hotfix mode (prepare-tag -> summary, skip validation)."""
    version = ctx.obj["version"]
    skip_commit_match = ctx.obj["skip_commit_match"]
    no_interactive = ctx.obj["no_interactive"]

    state_version = f"v{version}"
    state = load_or_create_state(state_version, git.get_current_commit())

    print(yellow("\n⚡ HOTFIX MODE: Skipping validation\n"))

    # Step 1: Prepare RC tag
    step_prepare_tag(version=version, state=state, no_interactive=no_interactive)

    # Create release
    step_release(version=version, skip_commit_match=skip_commit_match)


@app.command("release")
def cmd_release(
    ctx: typer.Context,
    job: Optional[str] = typer.Option(
        None,
        "--job",
        help="Filter validation jobs by name",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry previously failed jobs (default: skip failed jobs)",
    ),
):
    """Full release pipeline (prepare-tag -> validation -> bug check -> release tag)."""
    version = ctx.obj["version"]
    skip_commit_match = ctx.obj["skip_commit_match"]
    no_interactive = ctx.obj["no_interactive"]

    state_version = f"v{version}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Step 1: Prepare RC tag (automatic - skips if already done)
    step_prepare_tag(version=version, state=state, no_interactive=no_interactive)

    # Step 2: Run validation
    step_job_validation(
        version=version, job=job, retry=retry, skip_commit_match=skip_commit_match, no_interactive=no_interactive
    )

    # Step 3: Check for blocking bugs
    step_bug_check(version=version, state=state, skip_commit_match=skip_commit_match, no_interactive=no_interactive)

    # Step 4: Create release
    step_release(version=version, skip_commit_match=skip_commit_match)


if __name__ == "__main__":
    # Set up unified logging (shared across all versions)
    setup_logging(log_file())
    # Default to 'validate' if no subcommand was provided
    has_command = any(arg in ["validate", "hotfix", "release"] for arg in sys.argv[1:])
    if not has_command:
        sys.argv.append("validate")
    app()
