#!/usr/bin/env -S uv run
# ruff: noqa: E501
"""
Push the local mettaai/metta:latest image to an ECR registry.
"""

import argparse
import subprocess
import sys

import metta.common.util.cli
import metta.common.util.fs
import metta.common.util.text_styles


def main():
    metta.common.util.fs.cd_repo_root()

    parser = argparse.ArgumentParser(description="Upload metta image to ECR")
    parser.add_argument("--local-image-name", default="mettaai/metta:latest")
    parser.add_argument("--remote-image-name", default="metta:latest")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--account-id", type=int, help="AWS account ID. If omitted, current account is used.")
    args = parser.parse_args()

    account_id = args.account_id or metta.common.util.cli.sh(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]
    )
    if not account_id:
        sys.exit("ERROR: Failed to determine ACCOUNT_ID")

    local_image = metta.common.util.text_styles.bold(args.local_image_name)
    remote_image = metta.common.util.text_styles.bold(args.remote_image_name)
    print(f"Uploading {local_image} to {remote_image}")
    print(f"Region: {metta.common.util.text_styles.bold(args.region)}")
    print(f"Account ID: {metta.common.util.text_styles.bold(account_id)}")
    print("")
    if not metta.common.util.cli.get_user_confirmation(
        "Images should normally be uploaded by CI. Do you want to proceed?"
    ):
        sys.exit(0)

    push_image(args.local_image_name, args.remote_image_name, args.region, account_id)


def push_image(local_image_name: str, remote_image_name: str, region: str, account_id: str) -> None:
    docker_pwd = metta.common.util.cli.sh(["aws", "ecr", "get-login-password", "--region", region])
    host = f"{account_id}.dkr.ecr.{region}.amazonaws.com"

    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", host],
        input=docker_pwd,
        text=True,
        check=True,
    )

    subprocess.check_call(["docker", "tag", local_image_name, f"{host}/{remote_image_name}"])
    subprocess.check_call(["docker", "push", f"{host}/{remote_image_name}"])

    print("✓ Push complete")


if __name__ == "__main__":
    main()
