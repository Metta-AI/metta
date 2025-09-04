#!/usr/bin/env python3

import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime
from typing import List

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO

REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
WORKFLOW_FILENAME = "checks.yml"
WORKFLOW_NAME = "Test and Benchmark"
RUN_LINT = "true"
RUN_TEST = "true"
RUN_BENCHMARK = "true"


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


def wait_for_run_completion(run_id: str) -> dict:
    while True:
        result = subprocess.run(
            ["gh", "run", "view", run_id, "--json", "status,conclusion,startedAt,updatedAt"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            time.sleep(5)
            continue  # try again
        data = json.loads(result.stdout)
        if data["status"] == "completed":
            return data
        time.sleep(10)


def duration_seconds(start: str, end: str) -> float:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end_time - start_time).total_seconds()


def trigger_all_runs(branches: List[str], repeats: int) -> dict[str, List[str]]:
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


def wait_for_all_runs(run_ids_by_branch: dict[str, List[str]]) -> dict[str, List[float]]:
    print("\n⏳ Waiting for all workflow runs to complete...")
    durations_by_branch = {branch: [] for branch in run_ids_by_branch}
    for branch, run_ids in run_ids_by_branch.items():
        for run_id in run_ids:
            print(f"🔍 Waiting on {branch} → run {run_id}")
            try:
                run = wait_for_run_completion(run_id)
                dur = duration_seconds(run["startedAt"], run["updatedAt"])
                durations_by_branch[branch].append(dur)
                print(f"✅ {branch} → {dur:.1f}s")
            except Exception as e:
                print(f"❌ Failed to get result for {branch} run {run_id}: {e}")
    return durations_by_branch


def summarize(durations_by_branch: dict):
    print("\n📊 Benchmark Summary (seconds):")
    print(f"{'Branch':<20} {'Min':>8} {'Mean':>8} {'Max':>8} {'Runs':>5}")
    print("-" * 50)
    for branch, times in durations_by_branch.items():
        if times:
            print(f"{branch:<20} {min(times):8.1f} {statistics.mean(times):8.1f} {max(times):8.1f} {len(times):>5}")
        else:
            print(f"{branch:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GitHub workflow across branches.")
    parser.add_argument("branches", nargs="+", help="Branches to benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=10, help="Number of runs per branch")
    args = parser.parse_args()

    triggered = trigger_all_runs(args.branches, args.repeats)
    durations = wait_for_all_runs(triggered)
    summarize(durations)
