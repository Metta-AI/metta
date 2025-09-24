#!/usr/bin/env -S uv run

import argparse
import os
import subprocess


def main():
    """
    Create ~/.netrc and ~/.metta/observatory_tokens.yaml files with the given credentials.

    We pass the secrets to the job via environment variables, but we need to create the files for the job to use them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-password", type=str, required=True)
    parser.add_argument("--observatory-token", type=str)
    parser.add_argument("--profile", type=str)
    args = parser.parse_args()

    if args.profile:
        subprocess.run(["uv", "run", "--active", "metta", "configure", "--profile", args.profile, "--non-interactive"])

    if args.wandb_password:
        if os.path.exists(os.path.expanduser("~/.netrc")):
            print("~/.netrc already exists")
        else:
            with open(os.path.expanduser("~/.netrc"), "w") as f:
                f.write(f"machine api.wandb.ai\n  login user\n  password {args.wandb_password}\n")
            os.chmod(os.path.expanduser("~/.netrc"), 0o600)  # Restrict to owner read/write only

    if args.observatory_token:
        if os.path.exists(os.path.expanduser("~/.metta/observatory_tokens.yaml")):
            print("~/.metta/observatory_tokens.yaml already exists")
        else:
            os.makedirs(os.path.expanduser("~/.metta"), exist_ok=True)
            with open(os.path.expanduser("~/.metta/observatory_tokens.yaml"), "w") as f:
                f.write(f"https://api.observatory.softmax-research.net: {args.observatory_token}\n")
                f.write(f"https://observatory.softmax-research.net/api: {args.observatory_token}\n")
            print("~/.metta/observatory_tokens.yaml created")


if __name__ == "__main__":
    main()
