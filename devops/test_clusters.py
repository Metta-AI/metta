#!/usr/bin/env -S uv run
"""Test cluster configurations in parallel.

Usage:
    # Test default matrix (1, 2, 4 nodes)
    ./devops/test_clusters.py

    # Test specific node configs
    ./devops/test_clusters.py --nodes 1 2 8

    # Test with timeout conditions
    ./devops/test_clusters.py --test-timeouts
"""

from __future__ import annotations

import argparse

from devops.job_dispatcher import JobDispatcher
from devops.job_runner import RemoteJob

DEFAULT_MODULE = "experiments.recipes.arena_basic_easy_shaped.train"


def main():
    parser = argparse.ArgumentParser(
        description="Test cluster configurations in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Node configurations to test (default: 1 2 4)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="GPUs per node (default: 4)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total timesteps (default: 50000)",
    )

    parser.add_argument(
        "--test-timeouts",
        action="store_true",
        help="Test timeout conditions (heartbeat, runtime)",
    )

    parser.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"Module to test (default: {DEFAULT_MODULE})",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Job timeout in seconds (default: 3600)",
    )

    parser.add_argument(
        "--name",
        default="cluster_test",
        help="Name for this test run",
    )

    args = parser.parse_args()

    # Define test conditions
    conditions = {
        "normal": {
            "args": [f"trainer.total_timesteps={args.timesteps}"],
            "description": "Normal completion",
        },
    }

    if args.test_timeouts:
        conditions["heartbeat"] = {
            "args": ["-hb", "1"],  # 1 second heartbeat timeout
            "description": "Heartbeat timeout",
        }
        conditions["runtime"] = {
            "args": ["-t", "0.03"],  # 1.8 minutes
            "description": "Runtime timeout",
        }

    # Show what we'll test
    total_jobs = len(args.nodes) * len(conditions)
    print("=" * 80)
    print(f"Cluster Test Runner: {args.name}")
    print("=" * 80)
    print(f"Node configs: {args.nodes}")
    print(f"Conditions:   {list(conditions.keys())}")
    print(f"Total jobs:   {total_jobs}")
    print(f"Config:       {args.gpus} GPUs per node")
    print()

    # Create dispatcher
    dispatcher = JobDispatcher(name=args.name)

    # Create jobs for each combination
    for nodes in args.nodes:
        for cond_name, cond_info in conditions.items():
            job_name = f"{nodes}n_{cond_name}"

            base_args = ["--no-spot", f"--gpus={args.gpus}", "--nodes", str(nodes)]

            job = RemoteJob(
                name=job_name,
                module=args.module,
                args=cond_info["args"],
                base_args=base_args,
                timeout_s=args.timeout,
            )

            dispatcher.add_job(job)
            print(f"  • {job_name}: {cond_info['description']}")

    print()

    # Run all jobs
    print("Submitting jobs...")
    dispatcher.run_all()

    # Wait for completion
    print()
    results = dispatcher.wait_all(timeout_s=args.timeout)

    # Print summary
    dispatcher.print_summary()

    # Exit with error if any failed
    failed = sum(1 for r in results.values() if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit(main())
