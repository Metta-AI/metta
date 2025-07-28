#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

from metta.common.util.stats_client_cfg import get_machine_token
from metta.setup.utils import error, info


def main():
    parser = argparse.ArgumentParser(description="Launch Observatory locally with the correct token and backend URL")
    parser.add_argument(
        "--backend",
        choices=["local", "prod"],
        default="local",
        help="Backend to connect to (default: local)",
    )
    args = parser.parse_args()

    # Set environment variables based on backend
    env = os.environ.copy()

    if args.backend == "local":
        env["VITE_API_URL"] = "http://localhost:8000"
    elif args.backend == "prod":
        env["VITE_API_URL"] = "https://api.observatory.softmax-research.net"
        if token := get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token

    observatory_dir = Path(__file__).parent

    # Run npm dev
    info(f"Starting Observatory with backend: {args.backend}")
    info(f"API URL: {env.get('VITE_API_URL')}")
    if "VITE_AUTH_TOKEN" in env:
        info("Auth token: [CONFIGURED]")

    try:
        subprocess.run(["npm", "run", "dev"], env=env, check=True, cwd=observatory_dir)
    except subprocess.CalledProcessError as e:
        error(f"Error running npm dev: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        error("\nObservatory shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
