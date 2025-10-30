#!/usr/bin/env python3
"""Stable Release System - CLI

Commands:
  validate              Run validation (prepare-tag -> validation -> release)
  hotfix                Hotfix mode (prepare-tag -> release, skip validation)
  release               Create release (prepare-tag -> validation -> bug check -> release tag)

Options:
  --version X           Use specific version
  --new                 Force new release (ignore in-progress state)
  --task PATTERN        Filter validation tasks (validate mode only)
  --retry-failed        Retry failed tasks (validate mode only)
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
from devops.stable.display import check_task_passed, format_training_job_section
from devops.stable.runner import TaskRunner
from devops.stable.state import (
    ReleaseState,
    get_most_recent_state,
    load_or_create_state,
    load_state,
    save_state,
)
from devops.stable.tasks import get_all_tasks
from metta.common.util.fs import get_repo_root
from metta.common.util.text_styles import bold, cyan, green, yellow
from metta.jobs.job_manager import JobManager

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
    return any(g.get("step") == step_name and g.get("passed") for g in state.gates)


def mark_step_complete(state: ReleaseState | None, step_name: str) -> None:
    """Mark a pipeline step as completed in state."""
    if not state:
        return
    state.gates.append(
        {
            "step": step_name,
            "passed": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_state(state)


def get_user_confirmation(prompt: str) -> bool:
    """Get yes/no confirmation from user."""
    while True:
        try:
            response = input(f"{prompt} [y/N] ").strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no", ""):
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except (EOFError, KeyboardInterrupt):
            return False


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


def get_job_manager() -> JobManager:
    """Get JobManager instance for release validation (uses JobManager defaults)."""
    base_dir = get_repo_root() / "devops/stable/state"
    log_file = base_dir / "job_manager.log"

    # Set up file logging
    setup_logging(log_file)

    return JobManager(base_dir=base_dir)


def load_state_or_exit(version: str, step_name: str) -> ReleaseState:
    """Load release state or exit with error message."""
    state_version = f"v{version}"
    state = load_state(state_version)
    if not state:
        print(f"❌ No state found for version {version}")
        print(f"   Run 'validate' before '{step_name}'")
        sys.exit(1)
    return state


def verify_on_rc_commit(version: str, step_name: str) -> str:
    """Verify we're on the commit that the RC tag points to.

    Args:
        version: Release version
        step_name: Name of current step (for error messages)

    Returns:
        RC commit SHA

    Raises:
        SystemExit: If not on RC commit or RC tag doesn't exist
    """
    rc_tag_name = f"v{version}-rc"

    # Get current commit
    current_commit = git.get_current_commit()

    # Get RC tag commit
    try:
        rc_commit = git.run_git("rev-list", "-n", "1", rc_tag_name).strip()
    except git.GitError:
        print(f"❌ RC tag {rc_tag_name} not found")
        print("   Run 'prepare-tag' step first to create the RC tag")
        sys.exit(1)

    # Verify we're on the RC commit
    if current_commit != rc_commit:
        print(f"❌ ERROR: Not on RC commit for {step_name}")
        print(f"   Current commit:  {current_commit}")
        print(f"   RC tag commit:   {rc_commit} ({rc_tag_name})")
        print("\n   To fix, checkout the RC commit:")
        print(f"   git checkout {rc_tag_name}")
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
#   - Tracks individual validation task execution
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
#   - JobMonitor displays live status during task execution
#   - Monitor updates in-place (preserves step headers)
#   - Final summary shows all artifacts (WandB, checkpoints)
#   - Script only prints simple verdict after monitor summary
# ============================================================================


def step_prepare_tag(version: str, state: Optional[ReleaseState] = None, **_kwargs) -> None:
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
        if not get_user_confirmation("Delete existing tag and continue?"):
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


def step_bug_check(version: str, state: Optional[ReleaseState] = None, **_kwargs) -> None:
    """Check for blocking bugs via Asana or manual confirmation."""
    print_step_header("Bug Status Check")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "bug check")
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

    # Asana check inconclusive - fall back to manual
    print("⚠️  Asana automation unavailable")
    print("\nManual verification required:")
    print("  1. Check Asana for blocking bugs")
    print("  2. Resolve or triage any blockers\n")

    if not get_user_confirmation("Bug check PASSED?"):
        print("❌ Bug check FAILED")
        sys.exit(1)

    print("✅ Bug check PASSED")
    mark_step_complete(state, "bug_check")


def step_task_validation(
    version: str,
    task: Optional[str] = None,
    retry: bool = False,
    **_kwargs,
) -> None:
    """Run validation tasks via JobManager.

    JobManager handles execution, state persistence, and metrics extraction.
    Monitor displays live status with in-place updates.
    """
    print_step_header("Task Validation")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "task validation")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and filter tasks
    state_version = f"v{version}"
    state = load_or_create_state(state_version, git.get_current_commit())
    all_tasks = get_all_tasks()

    if task:
        tasks = [t for t in all_tasks if task in t.name]
        if not tasks:
            print(f"❌ No tasks matching '{task}'")
            print(f"Available: {', '.join(t.name for t in all_tasks)}")
            sys.exit(1)
        print(f"Running: {len(tasks)} task(s) matching '{task}'\n")
    else:
        tasks = all_tasks
        print(f"Running: {len(tasks)} task(s)\n")

    # Run tasks via JobManager
    job_manager = get_job_manager()
    log_file = get_repo_root() / "devops/stable/state/job_manager.log"
    print(f"💡 Detailed logs: tail -f {log_file}\n")
    runner = TaskRunner(state=state, job_manager=job_manager, retry_failed=retry)
    runner.run_all(tasks)

    # Count results (distinguish failed vs skipped)
    passed = 0
    failed = 0
    skipped = 0

    for task in tasks:
        job_state = job_manager.get_job_state(f"{state_version}_{task.name}")
        if not job_state:
            skipped += 1
        elif check_task_passed(job_state):
            passed += 1
        else:
            failed += 1

    # Print verdict
    print()
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

    print(f"✅ All task validations PASSED ({passed}/{len(tasks)})")


def step_summary(version: str, **_kwargs) -> None:
    """Print validation summary and release notes template.

    Queries JobManager for task outcomes and metrics.
    """
    print_step_header("Release Summary")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "summary")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state version and JobManager
    state_version = f"v{version}"
    _ = load_state_or_exit(version, "summary")  # Verify state exists
    job_manager = get_job_manager()

    # Get git log since last stable release
    git_log = git.git_log_since("origin/stable")

    # Get all tasks to display results and collect training job info
    all_tasks = get_all_tasks()
    training_jobs = []  # Track training jobs separately

    # Print task results summary
    print("Task Results:")
    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            if check_task_passed(job_state):
                icon = "✅"
            else:
                icon = "❌"
            print(f"  {icon} {task.name}")
            if job_state.metrics:
                metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
                print(f"       Metrics: {metrics_str}")

            # Collect training job info
            if task.job_config.is_training_job:
                training_jobs.append((task.name, job_state))

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
        for task_name, job_state in training_jobs:
            formatted = format_training_job_section(task_name, job_state)
            print(formatted)
            print("")
    else:
        print("- No training jobs in this validation run")
    print("")


def step_release(version: str, **_kwargs) -> None:
    """Create release tag and release notes.

    Verifies all validation tasks passed before creating release.
    """
    print_step_header("Create Release")

    # Verify on RC commit
    rc_commit = verify_on_rc_commit(version, "release")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and JobManager
    state_version = f"v{version}"
    state = load_state_or_exit(version, "release")
    job_manager = get_job_manager()

    # Verify all tasks passed
    all_tasks = get_all_tasks()

    failed_tasks = []
    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if not job_state:
            failed_tasks.append(f"{task.name} (not run)")
        elif not check_task_passed(job_state):
            failed_tasks.append(task.name)

    if failed_tasks:
        print("❌ Cannot release with failed/incomplete tasks:")
        for name in failed_tasks:
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
    # Use task results already validated above
    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            if check_task_passed(job_state):
                icon = "✅"
            else:
                icon = "❌"
            release_notes_content += f"- {icon} {task.name}\n"
            if job_state.metrics:
                metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
                release_notes_content += f"  - Metrics: {metrics_str}\n"

            # Collect training jobs
            if task.job_config.is_training_job:
                training_jobs.append((task.name, job_state))

    release_notes_content += f"""
## Changes Since Last Stable Release

{git_log if git_log else "No commits found"}

## Training Jobs

"""
    if training_jobs:
        for task_name, job_state in training_jobs:
            formatted = format_training_job_section(task_name, job_state)
            release_notes_content += formatted + "\n\n"
    else:
        release_notes_content += "- No training jobs in this validation run\n"

    # Write release notes
    release_notes_path.write_text(release_notes_content)
    print(f"✅ Created release notes: {release_notes_path}")

    # Get the commit SHA from the RC tag (validation was run against this)
    rc_tag_name = f"v{version}-rc"
    try:
        # Get commit SHA that RC tag points to
        rc_commit = git.run_git("rev-list", "-n", "1", rc_tag_name).strip()
        print(f"RC tag {rc_tag_name} points to commit: {rc_commit}")
    except git.GitError as e:
        print(f"❌ Failed to find RC tag {rc_tag_name}: {e}")
        print("   Run prepare-tag step first")
        sys.exit(1)

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
    print("\nNext steps:")
    print("  1. Run announce to notify team")


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
_VERSION: str | None = None  # Set by callback


@app.callback()
def common(
    version: Optional[str] = typer.Option(None, "--version", help="Version number (overrides auto-continue behavior)"),
    new: bool = typer.Option(False, "--new", help="Force start a new release (ignore in-progress state)"),
):
    """Stable Release System - automated release validation and deployment."""
    global _VERSION
    _VERSION = resolve_version(version, new)

    print("=" * 80)
    print(bold(cyan(f"Stable Release System - Version {_VERSION}")))
    print("=" * 80)
    print("\nContacts:")
    for contact in CONTACTS:
        print(f"  - {contact}")
    print("")


@app.command("validate")
def cmd_validate(
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Filter validation tasks by name",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry previously failed tasks (default: skip failed tasks)",
    ),
):
    """Run validation pipeline (prepare-tag -> validation -> summary)."""
    state_version = f"v{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Step 1: Prepare RC tag (automatic - skips if already done)
    step_prepare_tag(version=_VERSION, state=state)

    # Step 2: Run validation
    step_task_validation(version=_VERSION, task=task, retry=retry)

    # Step 3: Show summary
    step_summary(version=_VERSION)


@app.command("hotfix")
def cmd_hotfix():
    """Hotfix mode (prepare-tag -> summary, skip validation)."""
    state_version = f"v{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())

    print(yellow("\n⚡ HOTFIX MODE: Skipping validation\n"))

    # Step 1: Prepare RC tag
    step_prepare_tag(version=_VERSION, state=state)

    # Create release
    step_release(version=_VERSION)


@app.command("release")
def cmd_release():
    """Create release (bug check -> release tag)."""
    state_version = f"v{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Final gate: check for blocking bugs
    step_bug_check(version=_VERSION, state=state)

    # Create release
    step_release(version=_VERSION)


if __name__ == "__main__":
    # Default to 'validate' if no subcommand was provided
    has_command = any(arg in ["validate", "hotfix", "release"] for arg in sys.argv[1:])
    if not has_command:
        sys.argv.append("validate")
    app()
