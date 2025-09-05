#!/usr/bin/env python3
"""Simple debug script to check GitHub Actions timing"""

import json
import subprocess
import sys
from datetime import datetime


def format_duration(seconds):
    """Format seconds as human-readable"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s}s"


def parse_time(timestamp):
    """Parse ISO timestamp"""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_debug.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]

    # Get basic run info
    print(f"🔍 Checking run {run_id}\n")

    result = subprocess.run(
        ["gh", "run", "view", run_id, "--json", "startedAt,updatedAt,createdAt,conclusion,status"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)

    run_data = json.loads(result.stdout)

    print("📊 Workflow Timing:")
    print(f"  Created:  {run_data['createdAt']}")
    print(f"  Started:  {run_data['startedAt']}")
    print(f"  Updated:  {run_data['updatedAt']}")
    print(f"  Status:   {run_data['status']}")
    print(f"  Conclusion: {run_data.get('conclusion', 'N/A')}")

    created = parse_time(run_data["createdAt"])
    started = parse_time(run_data["startedAt"])
    updated = parse_time(run_data["updatedAt"])

    print("\n⏱️  Durations:")
    print(f"  Queue time (created→started): {(started - created).total_seconds():.0f}s")
    print(
        f"  Run time (started→updated):   {(updated - started).total_seconds():.0f}s = {format_duration((updated - started).total_seconds())}"
    )

    # Get jobs
    result = subprocess.run(["gh", "run", "view", run_id, "--json", "jobs"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ Failed to get jobs: {result.stderr}")
        return

    jobs = json.loads(result.stdout)["jobs"]

    print(f"\n📋 Jobs ({len(jobs)} total):")

    # Find actual start/end excluding late jobs
    actual_start = None
    actual_end = None
    late_jobs = []

    for job in jobs:
        if job["status"] != "completed":
            continue

        start = parse_time(job["startedAt"]) if job.get("startedAt") else None
        end = parse_time(job["completedAt"]) if job.get("completedAt") else None

        if start:
            # Check if this job started way after others (likely a post-workflow job)
            if actual_start and (start - actual_start).total_seconds() > 300:  # 5+ minutes later
                late_jobs.append(job["name"])
                continue

            if not actual_start or start < actual_start:
                actual_start = start

            if end and (not actual_end or end > actual_end):
                actual_end = end

    # Print jobs
    for job in jobs:
        if job["status"] != "completed":
            print(f"\n  ❌ {job['name']} - {job['status']}")
            continue

        start = parse_time(job["startedAt"]) if job.get("startedAt") else None
        end = parse_time(job["completedAt"]) if job.get("completedAt") else None

        if start and end:
            duration = (end - start).total_seconds()
            late_marker = " ⏰ (late job)" if job["name"] in late_jobs else ""
            print(f"\n  ✅ {job['name']}{late_marker}")
            print(f"     Duration: {duration:.0f}s = {format_duration(duration)}")
            print(f"     Start: {job['startedAt']}")
            print(f"     End:   {job['completedAt']}")

    if actual_start and actual_end:
        actual_duration = (actual_end - actual_start).total_seconds()
        print("\n⏱️  Actual workflow duration (excluding late jobs):")
        print(f"  {actual_duration:.0f}s = {format_duration(actual_duration)}")
        print("  This should be closer to what GitHub UI shows")

    # Try to get unit test job details
    unit_test_job = None
    for job in jobs:
        if "unit test" in job["name"].lower():
            unit_test_job = job
            break

    if unit_test_job:
        print(f"\n🧪 Attempting to get steps for '{unit_test_job['name']}'...")
        job_id = unit_test_job["databaseId"]

        # Try the API call
        api_url = f"https://api.github.com/repos/{{owner}}/{{repo}}/actions/jobs/{job_id}"
        print(f"   Note: You'll need to use: gh api /repos/OWNER/REPO/actions/jobs/{job_id}")
        print("   Replace OWNER/REPO with your actual repository")

        # Get the repo from the workflow URL
        result = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner"], capture_output=True, text=True)

        if result.returncode == 0:
            try:
                repo_data = json.loads(result.stdout)
                repo = repo_data["nameWithOwner"]
                print(f"\n   Detected repo: {repo}")

                # Now try to get the steps
                result = subprocess.run(
                    ["gh", "api", f"/repos/{repo}/actions/jobs/{job_id}"], capture_output=True, text=True
                )

                if result.returncode == 0:
                    job_details = json.loads(result.stdout)
                    steps = job_details.get("steps", [])

                    print(f"\n   Found {len(steps)} steps:")
                    for i, step in enumerate(steps):
                        if step["status"] == "completed":
                            name = step.get("name", "Unknown")
                            started_at = step.get("started_at")
                            completed_at = step.get("completed_at")

                            if started_at and completed_at:
                                start = parse_time(started_at)
                                end = parse_time(completed_at)
                                duration = (end - start).total_seconds()
                                print(f"\n   [{i:2d}] {name}")
                                print(f"        Duration: {duration:.0f}s = {format_duration(duration)}")
                                print(f"        Status: {step['conclusion']}")
                else:
                    print(f"\n   ❌ Failed to get job details: {result.stderr}")

            except Exception as e:
                print(f"\n   ❌ Error: {e}")


if __name__ == "__main__":
    main()
