#!/usr/bin/env python3

import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO

REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
WORKFLOW_FILENAME = "checks.yml"
WORKFLOW_NAME = "Test and Benchmark"
RUN_LINT = "true"
RUN_TEST = "true"
RUN_BENCHMARK = "true"

# Configuration constants
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds
POLL_INTERVAL = 10  # seconds


class WorkflowRunError(Exception):
    """Custom exception for workflow run errors"""

    pass


def trigger_workflow(branch: str) -> str:
    run_id = str(uuid.uuid4())[:8]
    print(f"🚀 Triggering workflow on branch: {branch} (run_id={run_id})")
    result = subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            WORKFLOW_FILENAME,
            "--ref",
            branch,
            "-f",
            f"run_lint={RUN_LINT}",
            "-f",
            f"run_test={RUN_TEST}",
            "-f",
            f"run_benchmark={RUN_BENCHMARK}",
            "-f",
            f"run_id={run_id}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to trigger workflow: {result.stderr}")

    time.sleep(2)  # allow GitHub to register the workflow run

    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            WORKFLOW_NAME,
            "--branch",
            branch,
            "--limit",
            "1",
            "--json",
            "databaseId",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch run list: {result.stderr}")
    runs = json.loads(result.stdout)
    if not runs:
        raise RuntimeError("No runs found after triggering workflow")
    return str(runs[0]["databaseId"])


def wait_for_run_completion(run_id: str) -> Tuple[dict, str]:
    retries = 0
    while True:
        result = subprocess.run(
            ["gh", "run", "view", run_id, "--json", "status,conclusion,startedAt,updatedAt"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            retries += 1
            if retries > MAX_RETRIES:
                raise WorkflowRunError(f"Failed to get run status after {MAX_RETRIES} retries: {result.stderr}")
            print(f"⚠️  Failed to get run status (attempt {retries}/{MAX_RETRIES}): {result.stderr}")
            time.sleep(RETRY_DELAY)
            continue

        # Reset retry counter on successful API call
        retries = 0

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise WorkflowRunError(f"Failed to parse run status JSON: {e}") from e

        if data["status"] == "completed":
            conclusion = data.get("conclusion", "unknown")
            if conclusion != "success":
                raise WorkflowRunError(f"Workflow run completed with conclusion: {conclusion}")
            return data, conclusion

        time.sleep(POLL_INTERVAL)


def duration_seconds(start: str, end: str) -> float:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end_time - start_time).total_seconds()


def trigger_all_runs(branches: List[str], repeats: int) -> Dict[str, List[str]]:
    print("\n🚀 Triggering all workflow runs...")
    run_ids_by_branch = {branch: [] for branch in branches}
    for branch in branches:
        for i in range(repeats):
            print(f"▶️  Trigger {i + 1}/{repeats} for `{branch}`")
            try:
                run_id = trigger_workflow(branch)
                run_ids_by_branch[branch].append(run_id)
            except Exception as e:
                print(f"❌ Failed to trigger workflow on `{branch}`: {e}")
    return run_ids_by_branch


def wait_for_all_runs(run_ids_by_branch: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
    print("\n⏳ Waiting for all workflow runs to complete...")
    results_by_branch = {branch: {"successful": [], "failed": []} for branch in run_ids_by_branch}

    for branch, run_ids in run_ids_by_branch.items():
        for run_id in run_ids:
            print(f"🔍 Waiting on {branch} → run {run_id}")
            try:
                run, conclusion = wait_for_run_completion(run_id)
                dur = duration_seconds(run["startedAt"], run["updatedAt"])
                results_by_branch[branch]["successful"].append(dur)
                print(f"✅ {branch} → {dur:.1f}s (conclusion: {conclusion})")
            except WorkflowRunError as e:
                print(f"❌ {branch} run {run_id} failed: {e}")
                results_by_branch[branch]["failed"].append(run_id)
            except Exception as e:
                print(f"❌ Unexpected error for {branch} run {run_id}: {e}")
                results_by_branch[branch]["failed"].append(run_id)

    return results_by_branch


def summarize(results_by_branch: Dict[str, Dict[str, List]]):
    print("\n📊 Benchmark Summary (seconds):")
    print(f"{'Branch':<20} {'Min':>8} {'Mean':>8} {'Max':>8} {'Success':>8} {'Failed':>7}")
    print("-" * 60)

    for branch, results in results_by_branch.items():
        successful_times = results["successful"]
        failed_count = len(results["failed"])

        if successful_times:
            min_time = min(successful_times)
            mean_time = statistics.mean(successful_times)
            max_time = max(successful_times)
            success_count = len(successful_times)
            print(f"{branch:<20} {min_time:8.1f} {mean_time:8.1f} {max_time:8.1f} {success_count:>8} {failed_count:>7}")
        else:
            print(f"{branch:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8} {failed_count:>7}")

    # Print failed run details if any
    total_failed = sum(len(results["failed"]) for results in results_by_branch.values())
    if total_failed > 0:
        print(f"\n⚠️  Total failed runs: {total_failed}")
        for branch, results in results_by_branch.items():
            if results["failed"]:
                print(f"  {branch}: {', '.join(results['failed'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GitHub workflow across branches.")
    parser.add_argument("branches", nargs="+", help="Branches to benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=10, help="Number of runs per branch")
    args = parser.parse_args()

    triggered = trigger_all_runs(args.branches, args.repeats)
    results = wait_for_all_runs(triggered)
    summarize(results)
