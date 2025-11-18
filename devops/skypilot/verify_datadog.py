#!/usr/bin/env -S uv run
"""Verify Datadog agent is running in a SkyPilot job."""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import sky
import sky.jobs
import yaml

from devops.skypilot.utils.job_helpers import get_jobs_controller_name


def get_regions_from_yaml(yaml_path: Path) -> list[str]:
    content = yaml.safe_load(yaml_path.read_text())
    any_of = content.get("resources", {}).get("any_of", [])
    return sorted({entry["region"] for entry in any_of if "region" in entry})


def wait_for_job_running(job_id: int, max_wait_minutes: int = 10) -> bool:
    """Wait for job to be in RUNNING state."""
    print(f"Waiting for job {job_id} to be RUNNING...")
    start_time = time.time()
    timeout = max_wait_minutes * 60

    while time.time() - start_time < timeout:
        request_id = sky.jobs.queue(refresh=True, skip_finished=True, all_users=True, job_ids=[job_id])
        jobs = sky.get(request_id)
        if jobs:
            status = jobs[0].get("status")
            status_str = str(status).split(".")[-1] if status else "UNKNOWN"
            print(f"  Job status: {status_str}")
            if status == sky.jobs.ManagedJobStatus.RUNNING:
                return True
            elif status in (
                sky.jobs.ManagedJobStatus.SUCCEEDED,
                sky.jobs.ManagedJobStatus.FAILED,
                sky.jobs.ManagedJobStatus.CANCELLED,
            ):
                print(f"  Job ended with status: {status_str}")
                return False
        time.sleep(10)

    print(f"  Timeout waiting for job to be RUNNING")
    return False


def check_datadog_agent(job_id: int) -> int:
    """Check if Datadog agent is running in the job container."""
    print(f"Looking up cluster for job {job_id}...")

    # Get job info to find cluster name
    request_id = sky.jobs.queue(refresh=True, skip_finished=True, all_users=True, job_ids=[job_id])
    jobs = sky.get(request_id)
    if not jobs:
        print(f"❌ Job {job_id} not found")
        return 1

    job = jobs[0]
    job_name = job.get("job_name", "")

    # Use EC2 lookup (same as connect.py)
    REGIONS = get_regions_from_yaml(Path("devops/skypilot/config/skypilot_run.yaml"))
    instance = None
    for region in REGIONS:
        os.environ["AWS_REGION"] = region
        try:
            output = subprocess.check_output(
                [
                    "aws",
                    "ec2",
                    "describe-instances",
                    "--filters",
                    f"Name=tag:Name,Values=*-{job_id}-*",
                    "--query",
                    "Reservations[0].Instances[0].PublicDnsName",
                    "--output",
                    "text",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if output and not output.startswith("None"):
                instance = output.strip()
                break
        except subprocess.CalledProcessError:
            continue

    if not instance:
        print(f"❌ Could not find EC2 instance for job {job_id}")
        print(f"   Tried regions: {REGIONS}")
        return 1

    cluster_name = instance

    print(f"✓ Found EC2 instance: {cluster_name}")

    # Get job info for user_hash (already have it from above)
    user_hash = job["user_hash"]
    key_path = f"/home/ubuntu/.sky/clients/{user_hash}/ssh/sky-key"

    if not os.path.exists(key_path):
        print(f"❌ SSH key not found at {key_path}")
        return 1

    # Commands to check Datadog agent
    check_commands = [
        "echo '=== Checking Datadog agent process ==='",
        "ps aux | grep -E '[d]atadog-agent|agent.*run' || echo 'No datadog-agent process found'",
        "echo ''",
        "echo '=== Checking if agent binary exists ==='",
        "test -f /opt/datadog-agent/bin/agent/agent && echo 'Agent binary exists' || echo 'Agent binary NOT found'",
        "echo ''",
        "echo '=== Checking agent status ==='",
        "if [ -f /opt/datadog-agent/bin/agent/agent ]; then /opt/datadog-agent/bin/agent/agent status 2>&1 || echo 'Agent status check failed'; else echo 'Agent binary not found, cannot check status'; fi",
        "echo ''",
        "echo '=== Checking agent log file ==='",
        "if [ -f /tmp/datadog-agent.log ]; then echo 'Log file exists, last 20 lines:'; tail -20 /tmp/datadog-agent.log; else echo 'Log file not found'; fi",
        "echo ''",
        "echo '=== Checking for agent in setup/run logs ==='",
        "grep -i datadog /workspace/metta/devops/skypilot/config/skypilot_run.sh 2>/dev/null | head -5 || echo 'No datadog references in run script'",
    ]

    docker_command = "docker exec $(docker ps -q) bash -c " + shlex.quote(" && ".join(check_commands))
    ssh_command = f"ssh -i {key_path} -o StrictHostKeyChecking=no ubuntu@{cluster_name} {shlex.quote(docker_command)}"

    print("\n" + "="*60)
    print("Checking Datadog agent status in container...")
    print("="*60 + "\n")

    try:
        jobs_controller_name = get_jobs_controller_name()
        full_command = f"ssh -t {jobs_controller_name} {shlex.quote(ssh_command)}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=30)
    except (ValueError, Exception) as e:
        print(f"Warning: Could not connect via jobs controller ({e})")
        print("Attempting direct connection...")
        result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True, timeout=30)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    # Check if agent is running
    if "datadog-agent" in result.stdout or "agent.*run" in result.stdout:
        if "No datadog-agent process found" not in result.stdout:
            print("\n✅ Datadog agent process is RUNNING")
            return 0
        else:
            print("\n❌ Datadog agent process is NOT running")
            return 1
    else:
        print("\n❌ Could not determine agent status from output")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Datadog agent is running in a SkyPilot job")
    parser.add_argument("job_id", type=int, help="The job ID to check")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for job to be RUNNING")
    args = parser.parse_args()

    if not args.no_wait:
        if not wait_for_job_running(args.job_id):
            print(f"❌ Job {args.job_id} is not in RUNNING state")
            return 1

    return check_datadog_agent(args.job_id)


if __name__ == "__main__":
    sys.exit(main())

