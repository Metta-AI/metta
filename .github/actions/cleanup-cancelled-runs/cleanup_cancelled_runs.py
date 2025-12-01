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

import datetime
import os
import sys
from itertools import islice
from typing import Any, Iterable, Mapping, Sequence

from github import Github  # pyright: ignore[reportMissingImports]


def is_superseded_run(cancelled_run: Any, all_runs: Sequence[Any]) -> bool:
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


def parse_workflow_identifiers(raw_input: str) -> list[str]:
    """Parse workflow identifiers from comma/newline separated input."""
    identifiers: list[str] = []
    for line in raw_input.replace(",", "\n").splitlines():
        value = line.strip()
        if value:
            identifiers.append(value)
    return identifiers


def resolve_workflow(workflows: Sequence[Any], identifier: str) -> Any | None:
    """Find a workflow that matches the provided identifier."""
    workflow_file_input = identifier.strip()
    workflow_basename = os.path.basename(workflow_file_input)
    candidate_paths = {workflow_file_input}
    if not workflow_file_input.startswith(".github/"):
        candidate_paths.add(f".github/workflows/{workflow_file_input}")

    for wf in workflows:
        wf_path = wf.path
        wf_basename = os.path.basename(wf_path)
        if wf_path in candidate_paths or wf_basename == workflow_basename:
            return wf
    return None


def dedupe_workflows(workflows: Iterable[Any]) -> list[Any]:
    """Remove duplicate workflows while preserving order."""
    seen_ids: set[int] = set()
    result: list[Any] = []
    for wf in workflows:
        if wf.id in seen_ids:
            continue
        seen_ids.add(wf.id)
        result.append(wf)
    return result


def process_workflow(
    workflow: Any,
    *,
    dry_run: bool,
    workflow_limit: int,
    per_workflow_caps: Mapping[str, int],
    max_age_days: int,
    all_runs_limit: int = 100,
) -> tuple[int, int, int]:
    """Process a single workflow and return (deleted, superseded, cancelled)."""
    print(f"\nWorkflow: {workflow.path} (ID: {workflow.id})")

    cutoff_date: datetime.datetime | None = None
    if max_age_days > 0:
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=max_age_days)

    # Fetch runs and stop early if we hit one older than cutoff_date
    # Runs are in reverse chronological order (newest first)
    all_runs: list[Any] = []
    for run in islice(workflow.get_runs(), all_runs_limit):
        # If we have a cutoff date and this run is too old, stop fetching
        # (all subsequent runs will also be too old)
        if cutoff_date:
            created_at = run.created_at
            if created_at.tzinfo:
                created_at = created_at.astimezone(datetime.timezone.utc)
            else:
                created_at = created_at.replace(tzinfo=datetime.timezone.utc)
            if created_at < cutoff_date:
                # Found a run older than cutoff - can stop fetching
                break
        all_runs.append(run)

    cancelled_runs = [run for run in all_runs if run.conclusion == "cancelled"]
    if len(cancelled_runs) == 0:
        print("  No cancelled runs.")
        return (0, 0, 0)

    superseded_runs = []
    for run in cancelled_runs:
        if is_superseded_run(run, all_runs):
            superseded_runs.append(run)

    not_superseded_count = len(cancelled_runs) - len(superseded_runs)
    wf_basename = os.path.basename(workflow.path)
    effective_cap = per_workflow_caps.get(workflow.path, per_workflow_caps.get(wf_basename, workflow_limit))
    effective_cap = max(0, min(workflow_limit, effective_cap))
    runs_to_delete = superseded_runs[:effective_cap]
    if len(runs_to_delete) == 0:
        if superseded_runs:
            print(
                f"  Superseded runs found ({len(superseded_runs)}) but cap reached "
                f"(limit {effective_cap} of requested {workflow_limit})."
            )
        else:
            print("  No superseded runs within the recent window.")
        return (0, len(superseded_runs), len(cancelled_runs))

    print(
        f"  {'Would delete' if dry_run else 'Deleting'} "
        f"{len(runs_to_delete)} superseded run(s) "
        f"(cap {effective_cap}, cancelled={len(cancelled_runs)}, superseded={len(superseded_runs)}, "
        f"kept={not_superseded_count})"
    )
    deleted_count = 0
    for run in runs_to_delete:
        run_label = f"Run #{run.run_number} [{run.head_branch or 'unknown'}] {run.html_url}"
        try:
            if dry_run:
                print(f"      [DRY-RUN] {run_label}")
            else:
                print(f"      {run_label}")
                run.delete()
            deleted_count += 1
        except Exception as exc:  # pragma: no cover - API failures are logged
            print(f"      ⚠️ Failed to delete run #{run.run_number}: {exc}")

    return (deleted_count, len(superseded_runs), len(cancelled_runs))


def main():
    """Main entry point."""
    # Get inputs from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    workflow_file = os.environ.get("INPUT_WORKFLOW-FILE")
    max_deletions = int(os.environ.get("INPUT_MAX-DELETIONS", "10"))
    per_workflow_default = os.environ.get("INPUT_MAX-DELETIONS", "10")
    max_deletions_per_workflow = int(os.environ.get("INPUT_MAX-DELETIONS-PER-WORKFLOW", per_workflow_default))
    workflow_caps_raw = os.environ.get("INPUT_WORKFLOW-CAPS", "")
    max_age_days = int(os.environ.get("INPUT_MAX-AGE-DAYS", "30"))
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

    workflow_identifiers = parse_workflow_identifiers(workflow_file)
    process_all_workflows = any(identifier.lower() in {"*", "all"} for identifier in workflow_identifiers)
    if process_all_workflows:
        workflow_identifiers = []

    if not process_all_workflows and not workflow_identifiers:
        print("❌ Error: At least one workflow identifier must be provided")
        sys.exit(1)

    target_description = "all workflows" if process_all_workflows else ", ".join(workflow_identifiers)
    print(f"Cleaning up cancelled runs for: {target_description}")
    print(f"Repository: {github_repository}")
    print(f"Max deletions per run: {max_deletions}")
    print(f"Per-workflow deletion cap: {max_deletions_per_workflow}")
    if max_age_days > 0:
        print(f"Skipping cancelled runs older than {max_age_days} day(s)")

    # Initialize GitHub client
    from github import Auth  # pyright: ignore[reportMissingImports]

    g = Github(auth=Auth.Token(github_token))
    repo = g.get_repo(github_repository)

    print("Fetching workflow information...")
    workflows = list(repo.get_workflows())
    if not workflows:
        print("❌ Error: No workflows found in repository")
        sys.exit(1)

    if process_all_workflows:
        target_workflows = workflows
    else:
        matched = []
        for identifier in workflow_identifiers:
            workflow = resolve_workflow(workflows, identifier)
            if not workflow:
                print(f"❌ Error: Workflow {identifier} not found in repository")
                sys.exit(1)
            matched.append(workflow)
        target_workflows = matched

    target_workflows = dedupe_workflows(target_workflows)
    if not target_workflows:
        print("❌ Error: No workflows selected for processing")
        sys.exit(1)

    print(f"Processing {len(target_workflows)} workflow(s)")

    overall_deleted = 0
    overall_superseded = 0
    overall_cancelled = 0
    remaining_deletions = max_deletions
    caps_by_workflow: dict[str, int] = {}
    if workflow_caps_raw.strip():
        for line in workflow_caps_raw.replace(",", "\n").splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name or not value:
                continue
            try:
                caps_by_workflow[name] = int(value)
            except ValueError:
                print(f"⚠️ Ignoring invalid workflow cap entry '{line}'")

    for workflow in target_workflows:
        if remaining_deletions <= 0:
            print("Reached overall deletion limit; skipping remaining workflows.")
            break
        workflow_cap = min(max_deletions_per_workflow, remaining_deletions)
        if workflow_cap <= 0:
            continue
        deleted, superseded, cancelled = process_workflow(
            workflow,
            dry_run=dry_run,
            workflow_limit=workflow_cap,
            per_workflow_caps=caps_by_workflow,
            max_age_days=max_age_days,
        )
        remaining_deletions -= deleted
        overall_deleted += deleted
        overall_superseded += superseded
        overall_cancelled += cancelled

    print(
        f"\n===== Overall Summary =====\n"
        f"Workflows processed: {len(target_workflows)}\n"
        f"Cancelled runs inspected: {overall_cancelled}\n"
        f"Superseded runs identified: {overall_superseded}\n"
        f"Runs {'that would be deleted' if dry_run else 'deleted'}: {overall_deleted}"
    )

    # Set output for GitHub Actions
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"deleted-count={overall_deleted}\n")


if __name__ == "__main__":
    main()
