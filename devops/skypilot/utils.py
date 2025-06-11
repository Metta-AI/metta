import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import sky

from metta.util.colorama import blue, bold, cyan, green, red, use_colors, yellow
from metta.util.fs import cd_repo_root
from metta.util.git import (
    get_commit_message,
    get_current_commit,
    has_unstaged_changes,
    is_commit_pushed,
)


def print_tip(text: str):
    print(blue(text), file=sys.stderr)


def dashboard_url():
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


def validate_critical_files(cmd: str, task_args: List[str]) -> bool:
    """Validate that all critical files exist before launching the task."""
    critical_files = [
        "./devops/skypilot/config/sk_train.yaml",
        f"./devops/{cmd}.sh",
    ]

    if cmd == "train":
        critical_files.append("tools/train.py")
    elif cmd == "sweep":
        critical_files.append("tools/sweep.py")
    elif cmd == "evolve":
        critical_files.append("tools/evolve.py")

    # Check for configuration files mentioned in task arguments
    config_files_to_check = []
    for task_arg in task_args:
        # Check for agent configuration
        if task_arg.startswith("agent="):
            agent_value = task_arg.split("=", 1)[1]
            config_files_to_check.append(f"./configs/agent/{agent_value}.yaml")

        # Check for trainer configuration
        elif task_arg.startswith("trainer="):
            trainer_value = task_arg.split("=", 1)[1]
            config_files_to_check.append(f"./configs/trainer/{trainer_value}.yaml")

        # Check for environment configuration
        elif task_arg.startswith("trainer.env="):
            env_value = task_arg.split("=", 1)[1]
            config_files_to_check.append(f"./configs/{env_value}.yaml")

        # Check for evaluation configuration
        elif task_arg.startswith("trainer.eval="):
            eval_value = task_arg.split("=", 1)[1]
            config_files_to_check.append(f"./configs/eval/{eval_value}.yaml")

    # Add config files to critical files list
    critical_files.extend(config_files_to_check)

    divider_length = 60
    divider = "=" * divider_length
    print(f"\n{divider}")
    print(bold("Validating critical files..."))
    print(f"{divider}")

    missing_files = []
    found_files = []

    for file in critical_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(red(f"❌ Missing required file: {file}"))
        else:
            found_files.append(file)
            print(green(f"✅ Found: {file}"))

    if missing_files:
        print(red("\nValidation failed - missing required files:"))
        print(red("=" * 50))

        # Group suggestions by type
        suggestions = []
        current_dir = Path.cwd()

        # Check if we're in the wrong directory
        if any("devops" in f for f in missing_files):
            suggestions.append("• Ensure you're running from the repository root directory")
            suggestions.append(f"  Current directory: {current_dir}")

        # Check for config file issues
        missing_configs = [f for f in missing_files if f.startswith("./configs/")]
        if missing_configs:
            suggestions.append("• Check your configuration file paths:")
            for config in missing_configs:
                suggestions.append(f"  {config}")
                # Try to find similar files
                config_dir = Path(config).parent
                if config_dir.exists():
                    similar_files = list(config_dir.glob("*.yaml"))
                    if similar_files:
                        suggestions.append(f"    Available in {config_dir}: {[f.name for f in similar_files[:3]]}")

        # Check for tool files
        missing_tools = [f for f in missing_files if f.startswith("tools/")]
        if missing_tools:
            suggestions.append("• Missing tool files - check your repository structure")

        if suggestions:
            print(yellow("\nSuggestions:"))
            for suggestion in suggestions:
                print(yellow(suggestion))

        return False

    print(green("All critical files found. Validation successful."))
    return True


def validate_git_state(
    git_ref: Optional[str] = None, skip_validation: bool = False, skip_push_check: bool = False
) -> bool:
    """Validate git state and check for unstaged changes."""
    if has_unstaged_changes() and not skip_validation:
        print(red("Error: You have unstaged changes in your repository."))
        print(red("These changes will not be reflected in the cloud environment."))
        print(yellow("Commit or stash your changes, or use --skip-validation to bypass this check."))
        return False

    # Use current commit if no git_ref provided
    if not git_ref:
        git_ref = get_current_commit()
        if not git_ref:
            print(red("Error: Could not get current commit hash. Ensure you're in a git repository."))
            return False

    # Validate that the commit is pushed
    if not skip_push_check and not is_commit_pushed(git_ref):
        print(red(f"Error: Git commit {git_ref} has not been pushed."))
        print("Please push your commit or use --skip-push-check to bypass.")
        return False

    return True


def display_task_summary(
    task: sky.Task,
    cmd: str,
    task_args: List[str],
    git_ref: Optional[str] = None,
    timeout_hours: Optional[float] = None,
    copies: int = 1,
) -> None:
    """Display a comprehensive summary of the task that will be launched."""
    divider_length = 60
    divider = "=" * divider_length

    print(f"\n{divider}")
    print(bold("Task will be launched with the following details:"))
    print(f"{divider}")

    print(f"Task Name: {task.name}")
    print(f"Command: {cmd}")

    if copies > 1:
        print(f"Number of Copies: {copies}")

    # Display resource information
    if task.resources:
        resource = list(task.resources)[0]  # Get first resource
        if hasattr(resource, "accelerators") and resource.accelerators:
            gpu_info = []
            for gpu_type, count in resource.accelerators.items():
                gpu_info.append(f"{count}x {gpu_type}")
            print(f"GPUs: {', '.join(gpu_info)}")

        if hasattr(resource, "cpus") and resource.cpus:
            print(f"CPUs: {resource.cpus}")

        if hasattr(resource, "memory") and resource.memory:
            print(f"Memory: {resource.memory}GB")

        if hasattr(resource, "use_spot"):
            spot_status = "Yes" if resource.use_spot else "No"
            print(f"Spot Instances: {spot_status}")

    if task.num_nodes and task.num_nodes > 1:
        print(f"Number of Nodes: {task.num_nodes}")

    # Display timeout information
    if timeout_hours:
        timeout_str = format_timeout_display(timeout_hours)
        print(bold(yellow(f"AUTO-TERMINATION: Task will terminate after {timeout_str}")))
    else:
        print(yellow("NO TIMEOUT SET: Task will run until completion"))

    # Display git information
    if git_ref:
        print(f"Git Reference: {git_ref}")
        commit_message = get_commit_message(git_ref)
        if commit_message:
            first_line = commit_message.split("\n")[0]
            print(f"Commit Message: {yellow(first_line)}")

    print(f"{'-' * divider_length}")

    # Display task arguments
    if task_args:
        print(yellow("Task Arguments:"))
        for i, arg in enumerate(task_args):
            print(f"  {i + 1}. {cyan(arg)}")

    print(f"\n{divider}")


def format_timeout_display(timeout_hours: float) -> str:
    """Format timeout duration for display."""
    total_minutes = int(timeout_hours * 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours > 0:
        if minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def get_user_confirmation(dry_run: bool = False, skip_validation: bool = False) -> bool:
    """Get user confirmation before launching the task."""
    if dry_run:
        print(bold("DRY RUN - No task will be launched"))
        return False

    if not skip_validation:
        response = input("Should we launch this task? (Y/n): ")
        if response.lower() not in ["", "y", "yes"]:
            print(yellow("Task launch cancelled by user."))
            return False

    return True


def setup_launch_environment(use_color: bool = True) -> None:
    """Set up the launch environment with proper directory and color settings."""
    # Set up colors
    use_colors(sys.stdout.isatty() and use_color)

    # Ensure we're in the repository root
    cd_repo_root()


def validate_and_launch_task(
    task: sky.Task,
    cmd: str,
    task_args: List[str],
    git_ref: Optional[str] = None,
    timeout_hours: Optional[float] = None,
    copies: int = 1,
    dry_run: bool = False,
    skip_validation: bool = False,
    skip_push_check: bool = False,
    use_color: bool = True,
) -> bool:
    """
    Complete validation and launch workflow for a SkyPilot task.

    Returns True if task was successfully launched (or would be in dry run mode).
    """
    # Set up environment
    setup_launch_environment(use_color)

    # Validate critical files
    if not validate_critical_files(cmd, task_args):
        sys.exit(1)

    # Validate git state
    if not validate_git_state(git_ref, skip_validation, skip_push_check):
        return False

    # Display task summary
    display_task_summary(task, cmd, task_args, git_ref, timeout_hours, copies)

    # Get user confirmation
    if not get_user_confirmation(dry_run, skip_validation):
        return False

    # Launch the task(s)
    try:
        if copies == 1:
            launch_task(task, dry_run)
        else:
            for i in range(copies):
                copy_task = task.copy()
                copy_task.name = f"{task.name}_{i + 1}"
                launch_task(copy_task, dry_run)

        return True

    except Exception as e:
        print(red(f"Failed to launch task: {e}"))
        return False
