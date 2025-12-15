#!/usr/bin/env python3
"""Test script for Datadog metrics conversion.

Run this to test the job-to-metrics conversion logic without running actual jobs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from devops.stable.datadog_metrics import job_to_metrics, jobs_to_metrics
from devops.stable.runner import AcceptanceCriterion, Job, JobStatus


def create_mock_training_job(
    name: str = "test.stable.arena_basic_easy_shaped.train_100m",
    status: JobStatus = JobStatus.SUCCEEDED,
    remote_gpus: int = 1,
    remote_nodes: int = 1,
    metrics: dict[str, float] | None = None,
    acceptance_passed: bool = True,
) -> Job:
    """Create a mock training job for testing."""
    if metrics is None:
        metrics = {
            "overview/sps": 45000.0,
            "env_agent/heart.gained": 1.5,
        }
    return Job(
        name=name,
        cmd=["uv", "run", "./tools/run.py", "test"],
        timeout_s=7200,
        remote_gpus=remote_gpus,
        remote_nodes=remote_nodes,
        acceptance=[
            AcceptanceCriterion(metric="overview/sps", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
        status=status,
        exit_code=0 if status == JobStatus.SUCCEEDED else 1,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_s=3600.0,
        metrics=metrics,
        acceptance_passed=acceptance_passed if status == JobStatus.SUCCEEDED else None,
    )


def create_mock_eval_job(
    name: str = "test.stable.arena_basic_easy_shaped.eval_remote",
    status: JobStatus = JobStatus.SUCCEEDED,
    is_remote: bool = True,
    metrics: dict[str, float] | None = None,
) -> Job:
    """Create a mock eval job for testing."""
    if metrics is None:
        metrics = {
            "heart_delta_pct": 5.2,
        }
    return Job(
        name=name,
        cmd=["uv", "run", "./tools/run.py", "test"],
        timeout_s=3600,
        remote_gpus=4 if is_remote else None,
        remote_nodes=1 if is_remote else None,
        status=status,
        exit_code=0 if status == JobStatus.SUCCEEDED else 1,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_s=1800.0,  # 30 minutes
        metrics=metrics,
        acceptance_passed=True if status == JobStatus.SUCCEEDED else None,
    )


def main():
    """Test the metrics conversion with mock jobs."""
    print("=" * 60)
    print("Testing Datadog Metrics Conversion")
    print("=" * 60)

    # Test 1: Multigpu training job (success)
    print("\n1. Multigpu training job (success):")
    job1 = create_mock_training_job(
        name="test.stable.arena_basic_easy_shaped.train_100m",
        remote_gpus=1,
        remote_nodes=1,
    )
    metrics1 = job_to_metrics(job1)
    print(f"   Generated {len(metrics1)} metrics:")
    for m in metrics1:
        print(f"   - {m.name}: {m.value} (tags: {m.tags})")

    # Test 2: Multinode training job (success)
    print("\n2. Multinode training job (success):")
    job2 = create_mock_training_job(
        name="test.stable.learning_progress.train_2b",
        remote_gpus=4,
        remote_nodes=4,
        metrics={"overview/sps": 85000.0, "overview/shaped_sps": 90000.0, "env_agent/heart.gained": 2.0},
    )
    metrics2 = job_to_metrics(job2)
    print(f"   Generated {len(metrics2)} metrics:")
    for m in metrics2:
        print(f"   - {m.name}: {m.value} (tags: {m.tags})")

    # Test 3: Failed training job
    print("\n3. Failed training job:")
    job3 = create_mock_training_job(
        name="test.stable.arena_basic_easy_shaped.train_100m",
        status=JobStatus.FAILED,
        acceptance_passed=False,
    )
    metrics3 = job_to_metrics(job3)
    print(f"   Generated {len(metrics3)} metrics:")
    for m in metrics3:
        print(f"   - {m.name}: {m.value} (tags: {m.tags})")

    # Test 4: Remote eval job
    print("\n4. Remote eval job (success):")
    job4 = create_mock_eval_job(
        name="test.stable.arena_basic_easy_shaped.eval_remote",
        is_remote=True,
    )
    metrics4 = job_to_metrics(job4)
    print(f"   Generated {len(metrics4)} metrics:")
    for m in metrics4:
        print(f"   - {m.name}: {m.value} (tags: {m.tags})")

    # Test 5: Local eval job
    print("\n5. Local eval job (success):")
    job5 = create_mock_eval_job(
        name="test.stable.arena_basic_easy_shaped.eval_local",
        is_remote=False,
    )
    metrics5 = job_to_metrics(job5)
    print(f"   Generated {len(metrics5)} metrics:")
    for m in metrics5:
        print(f"   - {m.name}: {m.value} (tags: {m.tags})")

    # Test 6: All jobs together
    print("\n6. All jobs together:")
    all_jobs = {
        "job1": job1,
        "job2": job2,
        "job3": job3,
        "job4": job4,
        "job5": job5,
    }
    all_metrics = jobs_to_metrics(all_jobs)
    print(f"   Total: {len(all_metrics)} metrics from {len(all_jobs)} jobs")

    # Print JSON output
    print("\n" + "=" * 60)
    print("JSON Output (ready for Datadog):")
    print("=" * 60)
    print(json.dumps([m.to_dict() for m in all_metrics], indent=2, default=str))


if __name__ == "__main__":
    main()
