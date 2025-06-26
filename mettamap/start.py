#!/usr/bin/env -S uv run

import subprocess
from pathlib import Path

from metta.common.fs import cd_repo_root


def main():
    cd_repo_root()
    repo_root = Path.cwd()
    mettamap_dir = repo_root / "mettamap"

    print(f"Starting servers from repo root: {repo_root}")
    print(f"Mettamap directory: {mettamap_dir}")

    # Start the backend server in repo root
    backend_process = subprocess.Popen(
        ["uv", "run", "-m", "metta.map.server"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Start the frontend dev server in mettamap directory
    frontend_process = subprocess.Popen(
        ["pnpm", "dev"], cwd=mettamap_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    print("Both servers started. Press Ctrl+C to stop both.")

    try:
        # Monitor both processes and print their output
        while True:
            # Check if either process has terminated
            if backend_process.poll() is not None:
                print("Backend server terminated")
                break
            if frontend_process.poll() is not None:
                print("Frontend server terminated")
                break

            # Read output from both processes
            backend_output = backend_process.stdout.readline()
            if backend_output:
                print(f"[BACKEND] {backend_output.strip()}")

            frontend_output = frontend_process.stdout.readline()
            if frontend_output:
                print(f"[FRONTEND] {frontend_output.strip()}")

    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()

        # Wait for processes to terminate
        backend_process.wait()
        frontend_process.wait()

        print("Both servers stopped.")


if __name__ == "__main__":
    main()
