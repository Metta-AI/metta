#!/usr/bin/env python3
"""Stable Release System - CLI

Command-line interface for the release validation and deployment pipeline.

Usage:
  ./devops/stable/release_stable.py                        # Run full release pipeline
  ./devops/stable/release_stable.py all                    # Run full release pipeline (explicit)
  ./devops/stable/release_stable.py prepare-tag            # Create staging tag
  ./devops/stable/release_stable.py bugs                   # Check bug status
  ./devops/stable/release_stable.py task-validation --task ci  # Run CI task
  ./devops/stable/release_stable.py task-validation        # Run all validation tasks
  ./devops/stable/release_stable.py summary                # Show validation summary
  ./devops/stable/release_stable.py release                # Create release tag
  ./devops/stable/release_stable.py announce               # Show announcement template

Global options:
  --version X           # Use specific version
  --new                 # Force start a new release (ignore in-progress state)

Examples:
  ./devops/stable/release_stable.py --new all              # Start new release and run full pipeline
  ./devops/stable/release_stable.py --version 2025.10.09-1030 summary  # Show summary for specific version
  ./devops/stable/release_stable.py task-validation --task ci  # Run only CI tests
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
from metta.common.util.text_styles import bold, cyan, green

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


# ============================================================================
# Release Pipeline Steps
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


def step_bug_check(state: Optional[ReleaseState] = None, **_kwargs) -> None:
    """Step 2: Check for blocking bugs."""
    print("\n" + "=" * 60)
    print(bold("STEP 2: Bug Status Check"))
    print("=" * 60 + "\n")

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
    reeval: bool = False,
    **_kwargs,
) -> None:
    """Step 3: Run validation tasks.

    Args:
        version: Release version
        task_filter: Task name to run (metta_ci, arena_local_smoke, arena_single_gpu_100m, etc.)
                    or None to run all tasks
        reeval: If True, re-evaluate failed runs from existing logs instead of re-launching
    """
    print("\n" + "=" * 60)
    print(bold("STEP 3: Task Validation"))
    print("=" * 60 + "\n")

    # Load or create state
    state_version = f"release_{version}"
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

    # Create runner and run all tasks
    runner = TaskRunner(state)
    runner.run_all(tasks)

    # Print summary
    passed = sum(1 for r in state.results.values() if r.outcome == "passed")
    failed = sum(1 for r in state.results.values() if r.outcome == "failed")

    print("\n" + "=" * 80)
    print("Task Summary")
    print("=" * 80)
    print("\nResults:")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")

    print("\nDetailed Results:")
    for result in state.results.values():
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏸️"}.get(result.outcome or "", "❓")
        print(f"  {icon} {result.name:24} exit={result.exit_code:>3}")
        if result.metrics:
            metrics_str = "  ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            print(f"       Metrics: {metrics_str}")
        if result.logs_path:
            print(f"       Logs: {result.logs_path}")
        if result.job_id:
            print(f"       Job ID: {result.job_id}")

    print("=" * 80)

    if failed:
        print(f"\nState saved to: devops/stable/state/{state_version}.json")
        print("❌ Task validation FAILED")
        sys.exit(1)

    print("\n✅ All task validations PASSED")
    print(f"State saved to: devops/stable/state/{state_version}.json")


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
        print("Run task-validation first")
        sys.exit(1)

    # Extract training metrics from TRAIN tasks
    training_metrics = {}
    training_job_id = None
    for name, result in state.results.items():
        if "train" in name.lower() and result.metrics:
            training_metrics.update(result.metrics)
            if result.job_id:
                training_job_id = result.job_id

    # Format metrics for display
    sps_max = training_metrics.get("sps_max", "N/A")
    sps_last = training_metrics.get("sps_last", "N/A")

    # Get git log since last stable release
    git_log = git.git_log_since("origin/stable")

    # Print task results summary
    print("Task Results:")
    for name, result in state.results.items():
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

    # Load state to get validation results
    state_version = f"release_{version}"
    state = load_state(state_version)

    if not state:
        print(f"No state found for version {version}")
        print("Run task-validation first")
        sys.exit(1)

    # Verify all tasks passed
    failed = [name for name, result in state.results.items() if result.outcome == "failed"]
    if failed:
        print("Cannot release with failed tasks:")
        for name in failed:
            print(f"  ❌ {name}")
        sys.exit(1)

    # Extract metrics for release notes
    training_metrics = {}
    training_job_id = None
    for name, result in state.results.items():
        if "train" in name.lower() and result.metrics:
            training_metrics.update(result.metrics)
            if result.job_id:
                training_job_id = result.job_id

    git_log = git.git_log_since("origin/stable")

    # Create release notes
    release_notes_dir = Path("devops/stable/release-notes")
    release_notes_dir.mkdir(parents=True, exist_ok=True)
    release_notes_path = release_notes_dir / f"v{version}.md"

    release_notes_content = f"""# Release Notes - Version {version}

## Task Results Summary

"""
    for name, result in state.results.items():
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
    print("\nNext steps:")
    print("  1. Run announce to notify team")


# ============================================================================
# CLI Setup
# ============================================================================

app = typer.Typer(add_completion=False)
_VERSION: str | None = None  # Set by callback


def generate_version() -> str:
    """Generate version string from current date/time.

    Format: YYYY.MM.DD-HHMM (e.g., 2025.10.07-1430)
    """
    return datetime.now().strftime("%Y.%m.%d-%H%M")


def resolve_version(explicit: Optional[str], force_new: bool) -> str:
    """Determine version based on args and state."""
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
        if recent_state.released:
            v = generate_version()
            print(f"Previous release ({recent_version}) completed. Starting new: {v}")
            return v
        print(f"Continuing in-progress release: {recent_version}")
        return recent_version
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


@app.command("prepare-tag")
def cmd_prepare_tag():
    """Create and push staging tag."""
    state_version = f"release_{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())
    step_prepare_tag(version=_VERSION, state=state)


@app.command("bugs")
def cmd_bugs():
    """Check bug status in Asana."""
    state_version = f"release_{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())
    step_bug_check(state=state)


@app.command("task-validation")
def cmd_task_validation(
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Validation filter: ci | train_local | train_remote | train_remote_multigpu | evaluate",
    ),
    reeval: bool = typer.Option(
        False,
        "--reeval",
        help="Re-evaluate failed runs from existing logs instead of re-launching",
    ),
):
    """Run validation tasks."""
    step_task_validation(version=_VERSION, task_filter=task, reeval=reeval)


@app.command("summary")
def cmd_summary():
    """Show validation summary and release notes template."""
    step_summary(version=_VERSION)


@app.command("release")
def cmd_release():
    """Create release tag and notes."""
    step_release(version=_VERSION)


@app.command("all")
def cmd_all(
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Optional validation filter (applies to task-validation step in the full pipeline)",
    ),
    reeval: bool = typer.Option(
        False,
        "--reeval",
        help="Re-evaluate failed runs from existing logs instead of re-launching",
    ),
):
    """Run full release pipeline."""
    state_version = f"release_{_VERSION}"
    state = load_or_create_state(state_version, git.get_current_commit())

    # Run steps with state tracking
    step_prepare_tag(version=_VERSION, state=state)
    step_bug_check(state=state)
    step_task_validation(version=_VERSION, task_filter=task, reeval=reeval)
    step_summary(version=_VERSION)


if __name__ == "__main__":
    # Default to 'all' if no subcommand was provided
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
