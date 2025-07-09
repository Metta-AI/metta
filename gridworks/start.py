#!/usr/bin/env -S uv run

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from metta.common.util.fs import cd_repo_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    args = parser.parse_args()

    cd_repo_root()
    repo_root = Path.cwd()
    gridworks_dir = repo_root / "gridworks"

    print(f"Starting servers from repo root: {repo_root}")
    print(f"Gridworks frontend directory: {gridworks_dir}")

    # Ensure color output from child processes even when stdout is piped.
    env = os.environ.copy()
    if os.isatty(sys.stdout.fileno()):
        env["FORCE_COLOR"] = "1"
        # some screen space is used by the label prefix
        env["COLUMNS"] = str(shutil.get_terminal_size().columns - 11)

    # Start the frontend dev server in gridworks directory
    if args.dev:
        frontend_process = subprocess.Popen(
            ["pnpm", "dev"],
            cwd=gridworks_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    else:
        # Build the production bundle before starting the server.
        print("Building frontend…")
        build_completed = subprocess.run(
            ["pnpm", "build"],
            cwd=gridworks_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Echo build output to the parent process stdout.
        if build_completed.stdout:
            print(build_completed.stdout)

        if build_completed.returncode != 0:
            print("Frontend build failed – aborting.")
            sys.exit(build_completed.returncode)

        # Launch the production server.
        frontend_process = subprocess.Popen(
            ["pnpm", "start"],
            cwd=gridworks_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    # Start the backend server in repo root
    backend_process = subprocess.Popen(
        ["uv", "run", "-m", "metta.gridworks.server"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("Both servers started. Press Ctrl+C to stop both.")

    RESET = "\033[0m"
    COLORS = {
        "BACKEND": "\033[94m",  # Bright Blue
        "FRONTEND": "\033[95m",  # Bright Magenta
    }

    backend_ready = False
    frontend_ready = False
    browser_opened = False

    def try_start_browser():
        nonlocal browser_opened
        if not browser_opened and frontend_ready and backend_ready:
            print("Trying to start browser")
            webbrowser.open("http://localhost:3000")
            browser_opened = True

    def stream_output(proc: subprocess.Popen[str], label: str):
        """Continuously read lines from a subprocess and print them with a label.

        When both backend and frontend indicate they are ready, automatically open the browser.
        """
        nonlocal backend_ready, frontend_ready
        assert proc.stdout is not None  # for mypy/static checkers

        for line in iter(proc.stdout.readline, ""):
            if not line:
                continue

            # Print the raw line with a colored prefixed label.
            color = COLORS.get(label, "")
            prefix = f"{color}[{label}]{RESET} "
            print(f"{prefix}{line.rstrip()}")

            # Detect the readiness message from the frontend dev server.
            if label == "FRONTEND" and "ready in" in line.lower():
                frontend_ready = True
                print("Frontend ready")
                try_start_browser()
            if label == "BACKEND" and "Uvicorn running on" in line:
                backend_ready = True
                print("Backend ready")
                try_start_browser()

    # Start dedicated threads to capture stdout from each process.
    backend_thread = threading.Thread(target=stream_output, args=(backend_process, "BACKEND"), daemon=True)
    frontend_thread = threading.Thread(target=stream_output, args=(frontend_process, "FRONTEND"), daemon=True)

    backend_thread.start()
    frontend_thread.start()

    try:
        # Keep running until either process exits or the user interrupts.
        while True:
            if backend_process.poll() is not None:
                print("Backend server terminated")
                break
            if frontend_process.poll() is not None:
                print("Frontend server terminated")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        backend_process.terminate()
        frontend_process.terminate()

        backend_process.wait()
        frontend_process.wait()

        # Give threads a moment to flush remaining output
        backend_thread.join(timeout=1)
        frontend_thread.join(timeout=1)

        print("Both servers stopped.")


if __name__ == "__main__":
    main()
