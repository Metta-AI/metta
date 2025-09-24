#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Save key/value benchmark data to a file compatible with Bencher.
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone


def get_machine_info():
    """Get machine information for benchmark metadata."""
    return {
        "node": socket.gethostname(),
        "processor": "GitHub Actions Runner",
        "machine": "GitHub Actions Runner",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
    }


def get_commit_info():
    """Get commit information from GitHub environment."""
    github_sha = os.environ.get("GITHUB_SHA", "unknown")
    github_ref_name = os.environ.get("GITHUB_REF_NAME", "unknown")
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "id": github_sha,
        "time": current_time,
        "author_time": current_time,
        "dirty": False,
        "project": "mettagrid",
        "branch": github_ref_name,
    }


def create_benchmark_result(name: str, metrics: dict, group: str = "default"):
    """Create a benchmark result in pytest format."""
    # Extract duration from metrics, default to 0 if not found
    duration = float(metrics.get("duration", 0))

    return {
        "name": name,
        "fullname": name,
        "group": group,
        "params": {"user": "ci"},
        "stats": {
            "min": duration,
            "max": duration,
            "mean": duration,
            "stddev": 0,
            "rounds": 1,
            "median": duration,
            "iqr": 0,
            "q1": duration,
            "q3": duration,
            "iqr_outliers": 0,
            "stddev_outliers": 0,
            "outliers": "0;0",
            "ld15iqr": duration,
            "hd15iqr": duration,
            "ops": None,
            "total": duration,
            "iterations": 1,
        },
    }


def main():
    """Main entry point."""
    # Get inputs from environment variables
    name = os.environ.get("BENCHMARK_NAME", "")
    metrics_json = os.environ.get("BENCHMARK_METRICS", "{}")
    filename = os.environ.get("BENCHMARK_FILENAME", "benchmark_results.json")
    group = os.environ.get("BENCHMARK_GROUP", "default")

    if not name:
        print("Error: BENCHMARK_NAME environment variable not set")
        sys.exit(1)

    print(f"Saving benchmark data for: {name}")
    print(f"Metrics: {metrics_json}")

    # Parse metrics JSON
    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing metrics JSON: {e}")
        print(f"Invalid JSON: {metrics_json}")
        sys.exit(1)

    # Create benchmark data structure
    benchmark_data = {
        "machine_info": get_machine_info(),
        "commit_info": get_commit_info(),
        "benchmarks": [create_benchmark_result(name, metrics, group)],
    }

    # Write to file
    try:
        with open(filename, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"Benchmark results written to: {filename}")
    except Exception as e:
        print(f"Error writing benchmark file: {e}")
        sys.exit(1)

    # Debug: Print file contents
    print(f"DEBUG: Contents of {filename}:")
    with open(filename, "r") as f:
        print(f.read())

    # Validate the JSON file
    try:
        with open(filename, "r") as f:
            json.load(f)
        print("DEBUG: File contains valid JSON")
    except json.JSONDecodeError:
        print("WARNING: File contains invalid JSON")
        sys.exit(1)


if __name__ == "__main__":
    main()
