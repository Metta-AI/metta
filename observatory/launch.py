#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.constants import DEV_STATS_SERVER_URI, PROD_STATS_SERVER_URI
from metta.setup.utils import error, info


def main():
    parser = argparse.ArgumentParser(description="Launch Observatory frontend locally")
    parser.add_argument(
        "--backend",
        choices=["local", "prod"],
        default="local",
        help="Backend API to connect to (default: local at localhost:8000)",
    )
    args = parser.parse_args()

    env = os.environ.copy()

    if args.backend == "local":
        env["VITE_API_URL"] = DEV_STATS_SERVER_URI
        info("Connecting to local backend at localhost:8000")
        info("Make sure backend is running: docker compose -f app_backend/docker-compose.dev.yml up")
    else:
        env["VITE_API_URL"] = PROD_STATS_SERVER_URI
        if token := get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token
        info("Connecting to prod backend")

    info("Starting Observatory frontend")
    info(f"API URL: {env.get('VITE_API_URL')}")

    try:
        subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=Path(__file__).parent)
    except subprocess.CalledProcessError as e:
        error(f'Error running "pnpm run dev": {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nObservatory shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
