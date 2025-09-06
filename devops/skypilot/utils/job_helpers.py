import netrc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import sky
import sky.jobs
import sky.server.common
import wandb

import gitta as git
from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.text_styles import blue, bold, cyan, green, red, yellow


def get_devops_skypilot_dir() -> Path:
    return Path(__file__).parent.parent


def get_jobs_controller_name() -> str:
    job_clusters = sky.get(sky.status(all_users=True, cluster_names=["sky-jobs-controller*"]))
    if len(job_clusters) == 0:
        raise ValueError("No job controller cluster found, is it running?")
    return job_clusters[0]["name"]


def print_tip(text: str) -> None:
    print(blue(text), file=sys.stderr)


def launch_task(task: sky.Task) -> str:
    request_id = sky.jobs.launch(task)

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {yellow(f'sky api logs {short_request_id}')}")
    dashboard_url = sky.server.common.get_server_url() + "/dashboard/jobs"
    print(f"- Or, visit: {yellow(dashboard_url)}")

    return request_id


def check_git_state(commit_hash: str) -> str | None:
    error_lines = []

    has_changes, status_output = git.has_unstaged_changes()
    if has_changes:
        error_lines.append(red("❌ You have uncommitted changes that won't be reflected in the cloud job."))
        error_lines.append("Options:")
        error_lines.append("  - Commit: git add . && git commit -m 'your message'")
        error_lines.append("  - Stash: git stash")
        error_lines.append("\nDebug:\n" + status_output)
        return "\n".join(error_lines)

    if not git.is_commit_pushed(commit_hash):
        commit_display = commit_hash[:8]
        error_lines.append(
            red(f"❌ Commit {commit_display} hasn't been pushed and won't be reflected in the cloud job.")
        )
        error_lines.append("Options:")
        error_lines.append("  - Push: git push")
        return "\n".join(error_lines)

    return None


def check_config_files(cmd_args: list[str]) -> bool:
    """Check that config files referenced in arguments actually exist."""
    config_files_to_check = []

    # Mapping of argument prefix to config file path template
    config_mappings = {
        "agent=": "./configs/agent/{}.yaml",
        "trainer=": "./configs/trainer/{}.yaml",
        "trainer.curriculum=": "./configs/{}.yaml",
        "sim=": "./configs/sim/{}.yaml",
    }

    for task_arg in cmd_args:
        for prefix, path_template in config_mappings.items():
            if task_arg.startswith(prefix):
                value = task_arg.split("=", 1)[1]
                config_path = path_template.format(value)
                config_files_to_check.append((task_arg, config_path))
                break

    missing_files = []
    for arg, config_path in config_files_to_check:
        if not Path(config_path).exists():
            missing_files.append((arg, config_path))

    if missing_files:
        print(red("❌ Config files not found:"))
        for arg, path in missing_files:
            print(f"  {arg} -> {path}")

            # Try to suggest similar files
            config_dir = Path(path).parent
            if config_dir.exists():
                yaml_files = list(config_dir.glob("*.yaml"))
                if yaml_files:
                    suggestions = [f.stem for f in yaml_files[:3]]
                    print(f"    Available: {', '.join(suggestions)}")

        print("Check your argument spelling and file paths.")
        return False

    return True


def display_job_summary(
    job_name: str,
    cmd: str,
    task_args: list[str],
    commit_hash: str,
    git_ref: str | None = None,
    timeout_hours: float | None = None,
    task: sky.Task | None = None,
    **kwargs,
) -> None:
    """Display a summary of the job that will be launched."""
    divider_length = 60
    divider = blue("=" * divider_length)

    print(f"\n{divider}")
    print(bold(blue("Job details:")))
    print(f"{divider}")

    print(f"{bold('Name:')} {yellow(job_name)}")

    # Extract resource info from task if provided
    if task:
        if task.resources:
            resource = list(task.resources)[0]  # Get first resource option

            # GPU info
            if hasattr(resource, "accelerators") and resource.accelerators:
                gpu_info = []
                for gpu_type, count in resource.accelerators.items():
                    gpu_info.append(f"{count}x {gpu_type}")
                print(f"{bold('GPUs:')} {yellow(', '.join(gpu_info))}")

            # CPU info
            if hasattr(resource, "cpus") and resource.cpus:
                print(f"{bold('CPUs:')} {yellow(str(resource.cpus))}")

            # Spot instance info
            if hasattr(resource, "use_spot"):
                spot_status = "Yes" if resource.use_spot else "No"
                print(f"{bold('Spot Instances:')} {yellow(spot_status)}")

        # Node count
        if task.num_nodes and task.num_nodes > 1:
            print(f"{bold('Nodes:')} {yellow(str(task.num_nodes))}")

    # Display any additional job details from kwargs (excluding 'task')
    for key, value in kwargs.items():
        if value is not None and key != "task":
            # Convert snake_case to Title Case for display
            display_key = key.replace("_", " ").title()
            print(f"{bold(display_key + ':')} {yellow(str(value))}")

    # Display timeout information with prominence
    if timeout_hours:
        timeout_mins = int(timeout_hours * 60)
        hours = timeout_mins // 60
        mins = timeout_mins % 60

        if hours > 0:
            if mins == 0:
                timeout_str = f"{hours}h"
            else:
                timeout_str = f"{hours}h {mins}m"
        else:
            timeout_str = f"{mins}m"

        print(f"{bold('Auto-termination:')} {yellow(timeout_str)}")
    else:
        print(f"{bold('Auto-termination:')} {yellow('None')}")

    if git_ref:
        print(f"{bold('Git Reference:')} {yellow(git_ref)}")

    print(f"{bold('Commit Hash:')} {yellow(commit_hash)}")

    commit_message = git.get_commit_message(commit_hash)
    if commit_message:
        first_line = commit_message.split("\n")[0]
        print(f"{bold('Commit Message:')} {yellow(first_line)}")

    pr_info = git.get_matched_pr(commit_hash, REPO_SLUG)
    if pr_info:
        pr_number, pr_title = pr_info
        first_line = pr_title.split("\n")[0]
        print(f"{bold('PR:')} {yellow(f'#{pr_number} - {first_line}')}")
    else:
        print(f"{bold('PR:')} {red('Not an open PR HEAD')}")

    print(blue("-" * divider_length))
    print(f"\n{bold('Command:')} {yellow(cmd)}")

    if task_args:
        print(bold("Task Arguments:"))
        for i, arg in enumerate(task_args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                print(f"  {i + 1}. {yellow(key)}={cyan(value)}")
            else:
                print(f"  {i + 1}. {yellow(arg)}")

    print(f"\n{divider}")


def set_task_secrets(task: sky.Task) -> None:
    """Write job secrets to task envs."""
    # Note: we can't mount these with `file_mounts` because of skypilot bug with service accounts.
    # Also, copying the entire `.netrc` is too much (it could contain other credentials).

    wandb_password = netrc.netrc(os.path.expanduser("~/.netrc")).hosts["api.wandb.ai"][2]
    if not wandb_password:
        raise ValueError("Failed to get wandb password, run 'metta install' to fix")

    observatory_token = get_machine_token("https://api.observatory.softmax-research.net")
    if not observatory_token:
        observatory_token = ""  # we don't have a token in CI

    if not wandb.api.api_key:
        raise ValueError("Failed to get wandb api key, run 'metta install' to fix")

    task.update_secrets(
        dict(
            WANDB_API_KEY=wandb.api.api_key,
            WANDB_PASSWORD=wandb_password,
            OBSERVATORY_TOKEN=observatory_token,
        )
    )


def open_job_log_from_request_id(request_id: str, wait_seconds: float = 1.0) -> tuple[str | None, str]:
    """Launch job log in a subprocess from a request ID."""

    # Wait for the job to be registered
    time.sleep(wait_seconds)

    result = subprocess.run(["sky", "api", "logs", request_id], capture_output=True, text=True)

    if result.returncode == 0:
        output = result.stdout
        job_id_match = re.search(r"ID:\s*(\d+)", output)

        if job_id_match:
            job_id = job_id_match.group(1)
            print(green(f"Job submitted with ID: {job_id}"))

            print(output)

            print(f"\n{blue('Tailing job logs...')}")
            try:
                subprocess.run(["sky", "jobs", "logs", job_id])
            except KeyboardInterrupt:
                print("\n" + yellow("Stopped tailing logs"))

            return job_id, output
        else:
            print(yellow("Job ID not found in output"))
            return None, output
    else:
        error_msg = f"Error getting logs: {result.stderr}"
        print(red(error_msg))
        return None, error_msg


def get_request_id_from_launch_output(output: str) -> str | None:
    """looks for "Submitted sky.jobs.launch request: XXX" pattern in cli output"""
    request_match = re.search(r"Submitted sky\.jobs\.launch request:\s*([a-f0-9-]+)", output)

    if request_match:
        return request_match.group(1)

    # Fallback patterns
    request_id_match = re.search(r"request[_-]?id[:\s]+([a-f0-9-]+)", output, re.IGNORECASE)
    if request_id_match:
        return request_id_match.group(1)

    return None


def get_job_id_from_request_id(request_id: str, wait_seconds: float = 1.0) -> str | None:
    """Get job ID from a request ID by querying sky api logs."""
    time.sleep(wait_seconds)  # Wait for job to be registered

    try:
        result = subprocess.run(["sky", "api", "logs", request_id], capture_output=True, text=True)

        if result.returncode == 0:
            job_id_match = re.search(r"ID:\s*(\d+)", result.stdout)
            if job_id_match:
                return job_id_match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def tail_job_log(job_id: str, lines: int = 100) -> str | None:
    try:
        # Get the full logs
        cmd = ["sky", "jobs", "logs", job_id]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            # Split into lines and get the tail
            log_lines = result.stdout.strip().split("\n")
            if len(log_lines) > lines:
                # Return only the last N lines
                return "\n".join(log_lines[-lines:])
            else:
                # Return all lines if less than requested
                return result.stdout
        else:
            return f"Error getting logs: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Timeout getting logs (job may still be provisioning)"
    except Exception as e:
        return f"Error: {str(e)}"
