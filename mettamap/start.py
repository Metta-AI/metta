#!/usr/bin/env -S uv run

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from metta.common.fs import cd_repo_root


def main():
    cd_repo_root()
    repo_root = Path.cwd()
    mettamap_dir = repo_root / "mettamap"

    print(f"Starting servers from repo root: {repo_root}")
    print(f"Mettamap directory: {mettamap_dir}")

    # Ensure color output from child processes even when stdout is piped.
    env = os.environ.copy()
    if os.isatty(sys.stdout.fileno()):
        env["FORCE_COLOR"] = "1"
        # some screen space is used by the label prefix
        env["COLUMNS"] = str(shutil.get_terminal_size().columns - 11)

    # Start the backend server in repo root
    backend_process = subprocess.Popen(
        ["uv", "run", "-m", "metta.map.server"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Start the frontend dev server in mettamap directory
    frontend_process = subprocess.Popen(
        ["pnpm", "dev"],
        cwd=mettamap_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("Both servers started. Press Ctrl+C to stop both.")

    browser_opened = False

    RESET = "\033[0m"
    COLORS = {
        "BACKEND": "\033[94m",  # Bright Blue
        "FRONTEND": "\033[95m",  # Bright Magenta
    }

    def stream_output(proc: subprocess.Popen[str], label: str):
        """Continuously read lines from a subprocess and print them with a label.

        When the frontend (pnpm dev) indicates it is ready, automatically open the browser.
        """
        nonlocal browser_opened
        assert proc.stdout is not None  # for mypy/static checkers
        for line in iter(proc.stdout.readline, ""):
            if not line:
                continue

            # Print the raw line with a colored prefixed label.
            color = COLORS.get(label, "")
            prefix = f"{color}[{label}]{RESET} "
            print(f"{prefix}{line.rstrip()}")

            # Detect the readiness message from the frontend dev server.
            if label == "FRONTEND" and not browser_opened and "ready in" in line.lower():
                # Open the default browser pointing to the local dev server.
                webbrowser.open("http://localhost:3000")
                browser_opened = True

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
