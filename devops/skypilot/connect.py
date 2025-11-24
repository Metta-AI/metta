#!/usr/bin/env -S uv run

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import sky.jobs
import yaml

from devops.skypilot.utils.job_helpers import get_jobs_controller_name


def get_regions_from_yaml(yaml_path: Path) -> list[str]:
    content = yaml.safe_load(yaml_path.read_text())
    any_of = content.get("resources", {}).get("any_of", [])
    return sorted({entry["region"] for entry in any_of if "region" in entry})


def main():
    parser = argparse.ArgumentParser(description="Connect to a running skypilot job")
    parser.add_argument(
        "--mode",
        choices=["container", "host"],
        default="container",
        help="Whether to connect to the job container or the host machine where the job is running",
    )
    parser.add_argument("job_id", type=int, help="The job ID to connect to")
    args = parser.parse_args()

    job_id = args.job_id

    REGIONS = get_regions_from_yaml(Path("devops/skypilot/config/skypilot_run.yaml"))

    instance = None
    for region in REGIONS:
        os.environ["AWS_REGION"] = region
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
        )
        if output and not output.startswith("None"):
            instance = output.strip()
            break

    if not instance:
        raise ValueError(f"Could not find EC2 instance for job {job_id}")

    request_id = sky.jobs.queue(refresh=True, skip_finished=True, all_users=True, job_ids=[job_id])
    jobs = sky.get(request_id)
    if not jobs:
        raise ValueError(f"Job {job_id} not found")
    job = jobs[0]

    user_hash = job["user_hash"]

    if args.mode == "container":
        job_host_command = "docker exec -it $(docker ps -q) bash"
    elif args.mode == "host":
        job_host_command = "bash"
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    key_path = f"/home/ubuntu/.sky/clients/{user_hash}/ssh/sky-key"
    inner_ssh_command = shlex.join(["ssh", "-t", "-i", key_path, f"ubuntu@{instance}", job_host_command])

    try:
        jobs_controller_name = get_jobs_controller_name()
        full_command = shlex.join(["ssh", "-t", jobs_controller_name, inner_ssh_command])
        subprocess.run(full_command, shell=True, check=False)
    except (ValueError, Exception):
        direct_command = shlex.join(["ssh", "-t", "-i", key_path, f"ubuntu@{instance}", job_host_command])
        subprocess.run(direct_command, shell=True)


if __name__ == "__main__":
    main()
