#!/usr/bin/env python3
"""Stable Release System - CLI

Simple 3-mode interface for release pipeline.

Commands:
  validate              Run validation (prepare-tag -> validation -> summary)
  hotfix                Hotfix mode (prepare-tag -> summary, skip validation)
  release               Create release (bug check -> release tag)

Options:
  --version X           Use specific version
  --new                 Force new release (ignore in-progress state)
  --task PATTERN        Filter validation tasks (validate mode only)
  --retry-failed        Retry failed tasks (validate mode only)

Examples:
  # Normal release workflow
  ./devops/stable/release_stable.py validate     # 1. Validate
  ./devops/stable/release_stable.py release      # 2. Release

  # Hotfix workflow
  ./devops/stable/release_stable.py hotfix       # 1. Skip validation
  ./devops/stable/release_stable.py release      # 2. Release (still checks bugs)

  # Development
  ./devops/stable/release_stable.py validate --task ci  # Filter to CI only
  ./devops/stable/release_stable.py validate --retry-failed  # Retry failures
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

import gitta as git
from devops.stable.asana_bugs import check_blockers
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
# State management:
# - Each step checks if already completed via state.gates
# - Steps append gate dict on success: {"step": "name", "passed": True, "timestamp": "..."}
# - State persisted to devops/stable/state/{version}.json
# - Allows resuming after interruption or failure
#
# RC tag workflow:
# - prepare_tag creates v{version}-rc tag
# - All subsequent steps verify on RC commit (ensures validation consistency)
# - release creates final v{version} tag pointing to same commit
# ============================================================================


def step_prepare_tag(version: str, state: Optional[ReleaseState] = None, **_kwargs) -> None:
    """Step 1: Create staging tag to mark commit for validation."""
    tag_name = f"v{version}-rc"

    print("\n" + "=" * 60)
    print(bold(f"STEP 1: Prepare Staging Tag (v{version})"))
    print("=" * 60 + "\n")

    # Check if already completed
    if state and any(g.get("step") == "prepare_tag" and g.get("passed") for g in state.gates):
        print(green("✅ Staging tag step already completed (skipping)"))
        print(f"   Tag {tag_name} was created in previous run")
        return

    commit_sha = git.get_current_commit()
    print(f"Current commit: {commit_sha}")

    try:
        # Check if tag already exists
        existing = git.run_git("tag", "-l", tag_name)
        if existing.strip():
            print(f"⚠️  Tag {tag_name} already exists")
            # If state exists and this is a continuation, mark as complete and continue
            if state:
                print(green("   Tag exists from previous run - marking step as complete"))
                state.gates.append(
                    {
                        "step": "prepare_tag",
                        "passed": True,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )
                save_state(state)
                return
            # Otherwise, ask if we should recreate it
            if not get_user_confirmation("Delete existing tag and continue?"):
                sys.exit(1)
            git.run_git("tag", "-d", tag_name)
            # Try to delete from remote too (may fail if doesn't exist there)
            try:
                git.run_git("push", "origin", f":refs/tags/{tag_name}")
            except git.GitError:
                pass

        print(f"Creating staging tag: {tag_name}")
        # Create lightweight tag
        git.run_git("tag", tag_name)
        git.run_git("push", "origin", tag_name)
        print(green(f"✅ Staging tag {tag_name} created and pushed successfully"))

        # Mark as completed
        if state:
            state.gates.append(
                {
                    "step": "prepare_tag",
                    "passed": True,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                }
            )
            save_state(state)
    except git.GitError as e:
        print(f"Failed to create/push tag: {e}")
        sys.exit(1)


def step_bug_check(version: str, state: Optional[ReleaseState] = None, **_kwargs) -> None:
    """Step 2: Check for blocking bugs."""
    print("\n" + "=" * 60)
    print(bold("STEP 2: Bug Status Check"))
    print("=" * 60 + "\n")

    # Verify we're on the RC commit
    rc_commit = verify_on_rc_commit(version, "bug check")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Check if already completed
    if state and any(g.get("step") == "bug_check" and g.get("passed") for g in state.gates):
        print(green("✅ Bug check step already completed (skipping)"))
        return

    # Try automated Asana check
    result = check_blockers()

    if result is True:
        print("✅ Bug check PASSED - clear for release")
        if state:
            state.gates.append(
                {
                    "step": "bug_check",
                    "passed": True,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                }
            )
            save_state(state)
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

    if not get_user_confirmation("Have you completed the bug status check and is it PASSED?"):
        print("❌ Bug check FAILED - user indicated issues remain")
        sys.exit(1)

    print("✅ Bug check PASSED - user confirmed")
    if state:
        state.gates.append(
            {
                "step": "bug_check",
                "passed": True,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
        save_state(state)


def step_task_validation(
    version: str,
    task_filter: Optional[str] = None,
    retry_failed: bool = False,
    **_kwargs,
) -> None:
    """Step 3: Run validation tasks via JobManager.

    Tasks are submitted to JobManager which handles execution and state persistence.
    Results are queried on-demand from JobManager's SQLite DB, not cached in ReleaseState.

    Args:
        version: Release version
        task_filter: Task name to run (python_ci, arena_local_smoke, arena_single_gpu_100m, etc.)
                    or None to run all tasks
        retry_failed: If True, retry failed tasks; if False, skip them (default: False)
    """
    print("\n" + "=" * 60)
    print(bold("STEP 3: Task Validation"))
    print("=" * 60 + "\n")

    # Verify we're on the RC commit
    rc_commit = verify_on_rc_commit(version, "task validation")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load or create state
    state_version = f"v{version}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Get all tasks
    all_tasks = get_all_tasks()

    # Filter tasks if requested
    if task_filter:
        tasks = [t for t in all_tasks if task_filter in t.name]
        if not tasks:
            print(f"Error: No tasks found matching '{task_filter}'")
            print(f"Available tasks: {', '.join(t.name for t in all_tasks)}")
            sys.exit(1)
        print(f"Running tasks matching '{task_filter}'\n")
    else:
        tasks = all_tasks
        print("Running all tasks\n")

    # Initialize JobManager (shared DB, uses groups to filter by version)
    base_dir = get_repo_root() / "devops/stable"
    job_manager = JobManager(base_dir=base_dir, max_local_jobs=1, max_remote_jobs=4)

    # Create runner and run all tasks
    runner = TaskRunner(state=state, job_manager=job_manager, retry_failed=retry_failed)
    runner.run_all(tasks)

    # Print summary by querying JobManager
    passed = 0
    failed = 0
    job_states = []

    for task in tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            job_states.append((task, job_state))
            if job_state.exit_code == 0 and runner._passed(job_name, task):
                passed += 1
            else:
                failed += 1

    print("\n" + "=" * 80)
    print("Task Summary")
    print("=" * 80)
    print("\nResults:")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")

    print("\nDetailed Results:")
    for task, job_state in job_states:
        job_name = f"{state_version}_{task.name}"
        if runner._passed(job_name, task):
            icon = "✅"
        else:
            icon = "❌"
        print(f"  {icon} {task.name:24} exit={job_state.exit_code or 0:>3}")
        if job_state.metrics:
            metrics_str = "  ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
            print(f"       Metrics: {metrics_str}")
        if job_state.logs_path:
            print(f"       Logs: {job_state.logs_path}")
        if job_state.job_id:
            print(f"       Job ID: {job_state.job_id}")

    print("=" * 80)

    if failed:
        print(f"\nState saved to: devops/stable/state/{state_version}.json")
        print("❌ Task validation FAILED")
        sys.exit(1)

    print("\n✅ All task validations PASSED")
    print(f"State saved to: devops/stable/state/{state_version}.json")


def step_summary(version: str, **_kwargs) -> None:
    """Step 4: Print validation summary and release notes template.

    Queries JobManager for task outcomes and metrics (not cached in ReleaseState).
    """
    print("\n" + "=" * 60)
    print(bold("STEP 4: Release Summary"))
    print("=" * 60 + "\n")

    # Verify we're on the RC commit
    rc_commit = verify_on_rc_commit(version, "summary")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and JobManager
    state_version = f"v{version}"
    state = load_state(state_version)

    if not state:
        print(f"No state found for version {version}")
        print("Run task-validation first")
        sys.exit(1)

    # Initialize JobManager to query results
    base_dir = get_repo_root() / "devops/stable"
    job_manager = JobManager(base_dir=base_dir, max_local_jobs=1, max_remote_jobs=4)

    # Query JobManager for training metrics from all jobs in this version's group
    training_metrics = {}
    training_job_id = None
    all_jobs = job_manager.get_group_jobs(state_version)
    for job_name in all_jobs:
        if "train" in job_name.lower():
            job_state = job_manager.get_job_state(job_name)
            if job_state:
                training_metrics.update(job_state.metrics or {})
                if job_state.job_id:
                    training_job_id = job_state.job_id

    # Format metrics for display
    sps = training_metrics.get("overview/sps", "N/A")

    # Get git log since last stable release
    git_log = git.git_log_since("origin/stable")

    # Get all tasks to display results
    all_tasks = get_all_tasks()
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    # Print task results summary
    print("Task Results:")
    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            if runner._passed(job_name, task):
                icon = "✅"
            else:
                icon = "❌"
            print(f"  {icon} {task.name}")
            if job_state.metrics:
                metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
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
    print(f"- Training throughput (SPS): {sps}")
    print("")


def step_release(version: str, **_kwargs) -> None:
    """Create release tag and release notes.

    Verifies all validation tasks passed by querying JobManager before creating release.
    """
    print("\n" + "=" * 60)
    print(bold("Create Release"))
    print("=" * 60 + "\n")

    # Verify we're on the RC commit
    rc_commit = verify_on_rc_commit(version, "release")
    print(f"✅ Verified on RC commit: {rc_commit}\n")

    # Load state and JobManager
    state_version = f"v{version}"
    state = load_state(state_version)

    if not state:
        print(f"No state found for version {version}")
        print("Run task-validation first")
        sys.exit(1)

    # Initialize JobManager to query results
    base_dir = get_repo_root() / "devops/stable"
    job_manager = JobManager(base_dir=base_dir, max_local_jobs=1, max_remote_jobs=4)

    # Verify all tasks passed by querying JobManager
    from devops.stable.tasks import get_all_tasks

    all_tasks = get_all_tasks()
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    failed_tasks = []
    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if not job_state:
            failed_tasks.append(f"{task.name} (not run)")
        elif not runner._passed(job_name, task):
            failed_tasks.append(task.name)

    if failed_tasks:
        print("❌ Cannot release with failed/incomplete tasks:")
        for name in failed_tasks:
            print(f"  ❌ {name}")
        sys.exit(1)

    # Extract metrics for release notes from all jobs in this version's group
    training_metrics = {}
    training_job_id = None
    all_jobs = job_manager.get_group_jobs(state_version)
    for job_name in all_jobs:
        if "train" in job_name.lower():
            job_state = job_manager.get_job_state(job_name)
            if job_state:
                training_metrics.update(job_state.metrics or {})
                if job_state.job_id:
                    training_job_id = job_state.job_id

    git_log = git.git_log_since("origin/stable")

    # Create release notes
    release_notes_dir = Path("devops/stable/release-notes")
    release_notes_dir.mkdir(parents=True, exist_ok=True)
    release_notes_path = release_notes_dir / f"v{version}.md"

    release_notes_content = f"""# Release Notes - Version {version}

## Task Results Summary

"""
    # Query JobManager for task results
    all_tasks = get_all_tasks()
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    for task in all_tasks:
        job_name = f"{state_version}_{task.name}"
        job_state = job_manager.get_job_state(job_name)
        if job_state:
            if runner._passed(job_name, task):
                icon = "✅"
            else:
                icon = "❌"
            release_notes_content += f"- {icon} {task.name}\n"
            if job_state.metrics:
                metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in job_state.metrics.items())
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

    Format: YYYY.MM.DD-HHMM (e.g., 2025.10.07-1430)
    """
    return datetime.now().strftime("%Y.%m.%d-%H%M")


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
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry previously failed tasks (default: skip failed tasks)",
    ),
):
    """Run validation pipeline (prepare-tag -> validation -> summary)."""
    state_version = f"v{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Step 1: Prepare RC tag (automatic - skips if already done)
    step_prepare_tag(version=_VERSION, state=state)

    # Step 2: Run validation
    step_task_validation(version=_VERSION, task_filter=task, retry_failed=retry_failed)

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

    # Step 2: Show summary (no validation)
    step_summary(version=_VERSION)


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
