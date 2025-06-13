import re
import sys
from pathlib import Path

import sky
import sky.jobs
import sky.server.common

from metta.util.colorama import blue, bold, cyan, green, magenta, red, yellow
from metta.util.git import has_unstaged_changes, is_commit_pushed


def print_tip(text: str):
    print(blue(text), file=sys.stderr)


def dashboard_url() -> str:
    url = sky.server.common.get_server_url()
    # strip username and password from server_url
    url = re.sub("https://.*@", "https://", url)
    return url


def launch_task(task: sky.Task, dry_run=False):
    if dry_run:
        print_tip("DRY RUN.")
        print_tip("Tip: Pipe this command to `| yq -P .` to get the pretty yaml config.\n")
        print(task.to_yaml_config())
        return

    request_id = sky.jobs.launch(task)

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {bold(f'sky api logs {short_request_id}')}")
    print(f"- Or, visit: {bold(f'{dashboard_url()}/api/stream?request_id={short_request_id}')}")
    print("  - To sign in, use credentials from your ~/.skypilot/config.yaml file.")
    print(f"- To cancel the request, run: {bold(f'sky api cancel {short_request_id}')}")


def check_git_state(commit_hash: str) -> str | None:
    """Check that the commit has been pushed and there are no staged changes."""

    error_lines = []

    if has_unstaged_changes():
        error_lines.append(red("❌ You have uncommitted changes that won't be reflected in the cloud job."))
        error_lines.append("Options:")
        error_lines.append("  - Commit: git add . && git commit -m 'your message'")
        error_lines.append("  - Stash: git stash")
        return "\n".join(error_lines)

    if not is_commit_pushed(commit_hash):
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
                break  # Found a match, no need to check other prefixes

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
    git_ref: str | None = None,
    commit_message: str | None = None,
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

    # Display git information
    if git_ref:
        print(f"{bold('Git Reference:')} {yellow(git_ref)}")
        if commit_message:
            first_line = commit_message.split("\n")[0]
            print(f"{bold('Commit Message:')} {yellow(first_line)}")

    print(blue("-" * divider_length))
    print(f"{bold('Command:')} {magenta(cmd)}")

    if task_args:
        print(bold("Task Arguments:"))
        for i, arg in enumerate(task_args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                print(f"  {i + 1}. {yellow(key)}={cyan(value)}")
            else:
                print(f"  {i + 1}. {yellow(arg)}")

    print(f"\n{divider}")


def get_user_confirmation(prompt: str = "Should we proceed?") -> bool:
    """Get user confirmation before proceeding with an action."""

    response = input(f"{prompt} (Y/n): ").strip().lower()
    if response not in ["", "y", "yes"]:
        print(yellow("Action cancelled by user."))
        return False

    return True
