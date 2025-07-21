#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def get_workflow_runs(branch_name: str, workflow_name: str) -> list[str]:
    """Get workflow run IDs for a given branch and workflow name."""
    command = [
        "gh",
        "run",
        "list",
        "--branch",
        branch_name,
        "--workflow",
        workflow_name,
        "--json",
        "databaseId",
    ]
    print(f"Searching for runs with: `{' '.join(command)}`")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting workflow runs: {result.stderr}", file=sys.stderr)
        return []

    if not result.stdout.strip():
        return []

    try:
        runs = json.loads(result.stdout)
        run_ids = [str(run["databaseId"]) for run in runs]
        return run_ids
    except json.JSONDecodeError:
        print(f"Could not decode JSON from gh command. Output: {result.stdout}", file=sys.stderr)
        return []


def download_logs_and_artifacts(run_id: str):
    """Download logs and artifacts for a specific run ID."""
    output_dir = f"run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing run {run_id} into directory ./{output_dir}/")

    # Download logs
    log_file_path = os.path.join(output_dir, "run_log.txt")
    print(f"  Downloading logs for run {run_id} to {log_file_path}...")
    try:
        with open(log_file_path, "w") as f:
            subprocess.run(["gh", "run", "view", run_id, "--log"], stdout=f, text=True, check=True)
        print("  Logs downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  Failed to download logs for run {run_id}. Stderr: {e.stderr}", file=sys.stderr)

    # Download artifacts
    print(f"  Downloading artifacts for run {run_id} to {output_dir}...")
    result = subprocess.run(
        ["gh", "run", "download", run_id, "--dir", output_dir, "-p", "*"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  Artifacts downloaded successfully.")
        if result.stdout:
            print(result.stdout)
    else:
        # gh run download exits with 1 if no artifacts are found.
        if "no artifacts found" in result.stderr.lower():
            print("  No artifacts found for this run.")
        else:
            print(f"  Error downloading artifacts for run {run_id}:", file=sys.stderr)
            print(f"  Stdout: {result.stdout}", file=sys.stderr)
            print(f"  Stderr: {result.stderr}", file=sys.stderr)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download GitHub Actions workflow logs and artifacts for a branch.")
    parser.add_argument("branch_name", help="The name of the GitHub branch.")
    parser.add_argument("--workflow", default="Asana Integration", help="The name of the GitHub workflow.")

    args = parser.parse_args()

    try:
        # Check if gh is installed
        subprocess.run(["gh", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Error: `gh` command not found or not working. Please ensure the GitHub CLI is installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        current_repo_result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner"], capture_output=True, text=True, check=True
        )
        repo_info = json.loads(current_repo_result.stdout)
        print(f"Operating in repository: {repo_info['nameWithOwner']}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        print(
            "Error: Could not determine current repository. Make sure you are in a git repository with a remote on GitHub.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_ids = get_workflow_runs(args.branch_name, args.workflow)

    if not run_ids:
        print(f"No runs found for workflow '{args.workflow}' on branch '{args.branch_name}'.")
        return

    print(f"Found {len(run_ids)} run(s): {', '.join(run_ids)}")

    for run_id in run_ids:
        download_logs_and_artifacts(run_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
