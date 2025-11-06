#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys

import metta.app_backend.clients.base_client
import metta.common.util.constants
import metta.setup.utils


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
        env["VITE_API_URL"] = metta.common.util.constants.DEV_STATS_SERVER_URI
    elif args.backend == "prod":
        env["VITE_API_URL"] = metta.common.util.constants.PROD_STATS_SERVER_URI
        if token := metta.app_backend.clients.base_client.get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token

    observatory_dir = pathlib.Path(__file__).parent

    # Run pnpm dev
    metta.setup.utils.info(f"Starting Observatory with backend: {args.backend}")
    metta.setup.utils.info(f"API URL: {env.get('VITE_API_URL')}")
    if "VITE_AUTH_TOKEN" in env:
        metta.setup.utils.info("Auth token: [CONFIGURED]")

    try:
        subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=observatory_dir)
    except subprocess.CalledProcessError as e:
        metta.setup.utils.error(f'Error running "pnpm run dev": {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        metta.setup.utils.error("\nObservatory shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
