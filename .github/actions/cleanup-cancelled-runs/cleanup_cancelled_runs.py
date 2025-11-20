#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "PyGithub>=2.1.1",
# ]
# ///
"""
Cleanup Cancelled Runs Action

Deletes workflow runs that were cancelled specifically due to concurrency settings
(i.e., superseded by a newer run). Does NOT delete runs that were manually cancelled
or failed for other reasons.

Note: The /// script section above is PEP 723 inline script metadata.
It tells `uv run` what dependencies to install when running this script directly.
"""

import os
import sys

# pyright: ignore[reportMissingImports]
from github import Github


def is_superseded_run(cancelled_run, all_runs) -> bool:
    """
    Check if a cancelled run was superseded by a newer run.

    Heuristic: If there's any newer run (whether still running or already finished)
    on the same branch/ref that wasn't itself cancelled, we treat the older cancelled
    run as superseded by concurrency settings.

    For main branch, we match any run on main (including merge_queue synthetic branches)
    since they all represent commits to the same branch.
    """
    cancelled_branch = cancelled_run.head_branch
    cancelled_sha = cancelled_run.head_sha
    cancelled_created = cancelled_run.created_at  # PyGithub always returns datetime objects

    # For main branch, match any run on main (including merge_queue branches)
    # For other branches, match exact branch name or SHA
    is_main_branch = cancelled_branch == "main" or (
        cancelled_branch and cancelled_branch.startswith("gh-readonly-queue/main/")
    )

    # Find newer runs on the same branch/ref
    newer_runs = []
    for run in all_runs:
        if run.id == cancelled_run.id:
            continue

        # Match logic: for main, accept any main branch run; otherwise match exact branch/SHA
        run_branch = run.head_branch
        run_sha = run.head_sha

        if is_main_branch:
            # For main, match if the run is also on main (including merge_queue)
            run_is_main = run_branch == "main" or (run_branch and run_branch.startswith("gh-readonly-queue/main/"))
            if not run_is_main:
                continue
        else:
            # For non-main branches, match exact branch or SHA
            if (run_branch or run_sha) != (cancelled_branch or cancelled_sha):
                continue

        # PyGithub datetime objects can be compared directly
        if run.created_at > cancelled_created:
            newer_runs.append(run)

    # Treat as superseded if any newer run is still running/queued, or if it completed
    # (regardless of success) and wasn't itself cancelled. This catches the concurrency
    # chain even when the replacement run fails due to a legitimate error.
    has_newer_non_cancelled_run = any(
        run.status in ("in_progress", "queued") or (run.conclusion and run.conclusion != "cancelled")
        for run in newer_runs
    )

    return len(newer_runs) > 0 and has_newer_non_cancelled_run


def main():
    """Main entry point."""
    # Get inputs from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    workflow_file = os.environ.get("INPUT_WORKFLOW-FILE")
    max_deletions = int(os.environ.get("INPUT_MAX-DELETIONS", "10"))
    dry_run = os.environ.get("INPUT_DRY-RUN", "false").lower() == "true"

    if not github_token:
        print("❌ Error: GITHUB_TOKEN is required")
        sys.exit(1)

    if not github_repository:
        print("❌ Error: GITHUB_REPOSITORY is required")
        sys.exit(1)

    if not workflow_file:
        print("❌ Error: workflow-file input is required")
        sys.exit(1)

    if dry_run:
        print("🔍 DRY-RUN MODE: Will only log what would be deleted, not actually delete")

    print(f"Cleaning up cancelled runs for workflow: {workflow_file}")
    print(f"Repository: {github_repository}")
    print(f"Max deletions per run: {max_deletions}")

    # Initialize GitHub client
    # pyright: ignore[reportMissingImports]
    from github import Auth

    g = Github(auth=Auth.Token(github_token))
    repo = g.get_repo(github_repository)

    # Get workflow by file name
    print("Fetching workflow information...")
    workflows = repo.get_workflows()
    workflow = None
    for wf in workflows:
        if wf.path.endswith(workflow_file):
            workflow = wf
            break

    if not workflow:
        print(f"❌ Error: Workflow {workflow_file} not found in repository")
        sys.exit(1)

    print(f"Found workflow ID: {workflow.id}")

    # Get all recent runs to check for superseding runs
    print("Fetching recent workflow runs...")
    all_runs = list(workflow.get_runs()[:100])
    print(f"Found {len(all_runs)} recent workflow runs")

    # Get cancelled runs
    cancelled_runs = [run for run in all_runs if run.conclusion == "cancelled"]
    print(f"Found {len(cancelled_runs)} cancelled runs")

    if len(cancelled_runs) == 0:
        print("No cancelled runs to process")
        # Set output for GitHub Actions
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write("deleted-count=0\n")
        return

    # Filter to only superseded runs
    print("Identifying superseded runs...")
    superseded_runs = [run for run in cancelled_runs if is_superseded_run(run, all_runs)]

    print(f"Identified {len(superseded_runs)} as superseded (will delete)")
    print(f"Keeping {len(cancelled_runs) - len(superseded_runs)} cancelled runs (not superseded)")

    # Limit to max-deletions to prevent rate limits
    runs_to_delete = superseded_runs[:max_deletions]

    if len(runs_to_delete) == 0:
        print("No superseded runs to delete")
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write("deleted-count=0\n")
        return

    print(f"Will {'simulate deletion of' if dry_run else 'delete'} {len(runs_to_delete)} superseded run(s)")

    # Delete only superseded runs
    deleted_count = 0
    errors = []

    for run in runs_to_delete:
        try:
            if dry_run:
                print(f"[DRY-RUN] Would delete: {run.html_url} (run #{run.run_number})")
                deleted_count += 1
            else:
                print(f"Deleting superseded cancelled workflow run: {run.html_url} (run #{run.run_number})")
                run.delete()
                deleted_count += 1
                print(f"✓ Successfully deleted run #{run.run_number}")
        except Exception as e:
            error_msg = f"Failed to delete run #{run.run_number}: {str(e)}"
            print(f"⚠️ {error_msg}")
            errors.append(error_msg)

    if dry_run:
        print(f"\n✨ Dry-run complete: {deleted_count} superseded run(s) would be deleted")
    else:
        print(f"\n✨ Cleanup complete: {deleted_count} superseded run(s) deleted")

    if errors:
        print(f"⚠️ Encountered {len(errors)} error(s) during deletion (see logs above)")

    # Set output for GitHub Actions
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"deleted-count={deleted_count}\n")


if __name__ == "__main__":
    main()
