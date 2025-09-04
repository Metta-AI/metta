#!/usr/bin/env python3

import json
import statistics
import subprocess
import time
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
    print(f"\n🚀 Triggering workflow on branch: {branch}")
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
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to trigger workflow: {result.stderr}")

    # Wait a moment to allow the run to be registered
    time.sleep(3)

    # Fetch the latest run for the workflow and branch
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

    run_id = str(runs[0]["databaseId"])
    return run_id


def wait_for_run_completion(run_id: str) -> dict:
    print(f"⏳ Waiting for run {run_id} to complete...")
    while True:
        result = subprocess.run(
            ["gh", "run", "view", run_id, "--json", "status,conclusion,startedAt,updatedAt"],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        if data["status"] == "completed":
            break
        time.sleep(10)
    return data


def duration_seconds(start: str, end: str) -> float:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end_time - start_time).total_seconds()


def benchmark_branch(branch: str, repeats: int) -> List[float]:
    durations = []
    for i in range(repeats):
        print(f"\n▶️  Run {i + 1}/{repeats} for branch `{branch}`")
        run_id = trigger_workflow(branch)
        run_info = wait_for_run_completion(run_id)
        dur = duration_seconds(run_info["startedAt"], run_info["updatedAt"])
        durations.append(dur)
        print(f"✅ Completed in {dur:.1f} seconds")
    return durations


def summarize(durations_by_branch: dict):
    print("\n📊 Benchmark Summary (seconds):")
    print(f"{'Branch':<20} {'Min':>8} {'Mean':>8} {'Max':>8} {'Runs':>5}")
    print("-" * 50)
    for branch, times in durations_by_branch.items():
        print(f"{branch:<20} {min(times):8.1f} {statistics.mean(times):8.1f} {max(times):8.1f} {len(times):>5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GitHub workflow across branches.")
    parser.add_argument("branches", nargs="+", help="Branches to benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=10, help="Number of runs per branch")
    args = parser.parse_args()

    durations_by_branch = {}

    for branch in args.branches:
        try:
            durations_by_branch[branch] = benchmark_branch(branch, args.repeats)
        except Exception as e:
            print(f"❌ Error benchmarking {branch}: {e}")

    summarize(durations_by_branch)
