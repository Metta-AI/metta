#!/usr/bin/env -S uv run

import argparse
import os
import shlex
import subprocess

import sky.jobs

from devops.skypilot.utils import get_jobs_controller_name
from metta.common.util.text_styles import bold


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

    print("Looking up EC2 instance...")

    # must match sk_train.yaml
    REGIONS = ["us-east-1", "us-west-2"]

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

    print(f"Found EC2 instance: {instance}")

    print("Looking up skypilot job...")
    request_id = sky.jobs.queue(refresh=True, skip_finished=True, all_users=True)
    jobs = sky.get(request_id)
    job = next((job for job in jobs if job["job_id"] == job_id), None)
    if job is None:
        print(jobs[0])
        raise ValueError(f"Job {job_id} not found")

    user_hash = job["user_hash"]

    if args.mode == "container":
        job_host_command = "docker exec -it $(docker ps -q) bash"
    elif args.mode == "host":
        job_host_command = "bash"
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    key_path = f"/home/ubuntu/.sky/clients/{user_hash}/ssh/sky-key"
    inner_ssh_command = shlex.join(["ssh", "-t", "-i", key_path, f"ubuntu@{instance}", job_host_command])

    print("Looking up jobs controller...")
    jobs_controller_name = get_jobs_controller_name()

    full_command = shlex.join(["ssh", "-t", jobs_controller_name, inner_ssh_command])
    print(f"Connecting with: {bold(full_command)}")

    subprocess.run(full_command, shell=True)


if __name__ == "__main__":
    main()
