#!/usr/bin/env -S uv run
"""
Push the local mettaai/metta:latest image to an ECR registry.
"""

import argparse
import subprocess
import sys

from metta.common.util.cli import get_user_confirmation, sh
from metta.common.util.colorama import bold
from metta.common.util.fs import cd_repo_root


def main():
    cd_repo_root()

    parser = argparse.ArgumentParser(description="Upload metta image to ECR")
    parser.add_argument("--local-image-name", default="mettaai/metta:latest")
    parser.add_argument("--remote-image-name", default="metta:latest")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--account-id", type=int, help="AWS account ID. If omitted, current account is used.")
    args = parser.parse_args()

    account_id = args.account_id or sh(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"])
    if not account_id:
        sys.exit("ERROR: Failed to determine ACCOUNT_ID")

    host = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com"

    print(f"Uploading {bold(args.local_image_name)} to {bold(args.remote_image_name)}")
    print(f"Region: {bold(args.region)}")
    print(f"Account ID: {bold(account_id)}")
    print("")
    if not get_user_confirmation("Images should normally be uploaded by CI. Do you want to proceed?"):
        sys.exit(0)

    docker_pwd = sh(["aws", "ecr", "get-login-password", "--region", args.region])

    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", host],
        input=docker_pwd,
        text=True,
        check=True,
    )

    subprocess.check_call(["docker", "tag", args.local_image_name, f"{host}/{args.remote_image_name}"])
    subprocess.check_call(["docker", "push", f"{host}/{args.remote_image_name}"])

    print("✓ Push complete")


if __name__ == "__main__":
    main()
