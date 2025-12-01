#!/usr/bin/env python3

import argparse
import atexit
import base64
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.constants import DEV_STATS_SERVER_URI, PROD_STATS_SERVER_URI
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info

PROD_DB_RDS_HOST = "main-pg.c6puykw82uvz.us-east-1.rds.amazonaws.com"
PROD_DB_PROXY_LOCAL_PORT = 15432
PROD_DB_NAMESPACE = "observatory"
PROD_DB_PROXY_POD = "pg-proxy"


def _get_prod_db_uri() -> str:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "secret",
            "observatory-backend-env",
            "-n",
            PROD_DB_NAMESPACE,
            "-o",
            "jsonpath={.data.STATS_DB_URI}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    uri = base64.b64decode(result.stdout).decode()
    pattern = re.escape(PROD_DB_RDS_HOST) + r"(:\d+)?"
    return re.sub(pattern, f"localhost:{PROD_DB_PROXY_LOCAL_PORT}", uri)


def _ensure_proxy_pod() -> None:
    check = subprocess.run(
        ["kubectl", "get", "pod", PROD_DB_PROXY_POD, "-n", PROD_DB_NAMESPACE],
        capture_output=True,
    )
    if check.returncode != 0:
        info(f"Proxy pod '{PROD_DB_PROXY_POD}' not found in namespace '{PROD_DB_NAMESPACE}'")
        info("To create manually:")
        info(
            f"  kubectl run {PROD_DB_PROXY_POD} -n {PROD_DB_NAMESPACE} --image=alpine/socat "
            f"--restart=Never -- TCP-LISTEN:5432,fork,reuseaddr TCP:{PROD_DB_RDS_HOST}:5432"
        )
        info(f"Creating proxy pod {PROD_DB_PROXY_POD}...")
        subprocess.run(
            [
                "kubectl",
                "run",
                PROD_DB_PROXY_POD,
                "-n",
                PROD_DB_NAMESPACE,
                "--image=alpine/socat",
                "--restart=Never",
                "--",
                "TCP-LISTEN:5432,fork,reuseaddr",
                f"TCP:{PROD_DB_RDS_HOST}:5432",
            ],
            check=True,
        )
        info("Waiting for proxy pod to be ready...")
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--for=condition=Ready",
                f"pod/{PROD_DB_PROXY_POD}",
                "-n",
                PROD_DB_NAMESPACE,
                "--timeout=30s",
            ],
            check=True,
        )
    else:
        info(f"Using existing proxy pod '{PROD_DB_PROXY_POD}'")


def _start_port_forward() -> subprocess.Popen:
    info(f"Starting port-forward to {PROD_DB_PROXY_POD} on localhost:{PROD_DB_PROXY_LOCAL_PORT}...")
    proc = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "-n",
            PROD_DB_NAMESPACE,
            PROD_DB_PROXY_POD,
            f"{PROD_DB_PROXY_LOCAL_PORT}:5432",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError("Port-forward failed to start (port may be in use)")
    return proc


def _start_stats_server(env: dict[str, str]) -> subprocess.Popen:
    repo_root = get_repo_root()
    server_path = repo_root / "app_backend" / "src" / "metta" / "app_backend" / "server.py"
    info("Starting stats-server with prod database...")
    proc = subprocess.Popen(
        ["uv", "run", "python", str(server_path)],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError("Stats server failed to start")
    return proc


def main():
    parser = argparse.ArgumentParser(description="Launch Observatory locally with the correct token and backend URL")
    parser.add_argument(
        "--backend",
        choices=["local", "prod"],
        default="local",
        help="Backend to connect to (default: local)",
    )
    parser.add_argument(
        "--db",
        choices=["local", "prod"],
        default="local",
        help="Database to connect to (default: local). Starts stats-server with prod DB when set to prod",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    port_forward_proc = None
    stats_server_proc = None

    if args.backend == "local":
        env["VITE_API_URL"] = DEV_STATS_SERVER_URI

        if args.db == "prod":
            _ensure_proxy_pod()
            port_forward_proc = _start_port_forward()
            atexit.register(lambda: port_forward_proc.terminate() if port_forward_proc else None)
            env["STATS_DB_URI"] = _get_prod_db_uri()
            info(f"Using prod database via localhost:{PROD_DB_PROXY_LOCAL_PORT}")
            stats_server_proc = _start_stats_server(env)
            atexit.register(lambda: stats_server_proc.terminate() if stats_server_proc else None)

    elif args.backend == "prod":
        env["VITE_API_URL"] = PROD_STATS_SERVER_URI
        if token := get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token
        if args.db == "prod":
            info("Note: --db prod is ignored when --backend=prod (already using prod DB)")

    observatory_dir = Path(__file__).parent

    info(f"Starting Observatory with backend: {args.backend}")
    info(f"API URL: {env.get('VITE_API_URL')}")
    if "VITE_AUTH_TOKEN" in env:
        info("Auth token: [CONFIGURED]")
    if stats_server_proc:
        info("Stats server: running (with prod DB)")

    try:
        subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=observatory_dir)
    except subprocess.CalledProcessError as e:
        error(f'Error running "pnpm run dev": {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        error("\nObservatory shutdown")
        sys.exit(0)
    finally:
        if stats_server_proc:
            stats_server_proc.terminate()
        if port_forward_proc:
            port_forward_proc.terminate()


if __name__ == "__main__":
    main()
