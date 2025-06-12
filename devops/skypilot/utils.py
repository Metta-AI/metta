import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import sky

from metta.util.colorama import blue, bold, cyan, green, red, use_colors, yellow
from metta.util.fs import cd_repo_root


def print_tip(text: str):
    print(blue(text), file=sys.stderr)


def dashboard_url():
    url = sky.server.common.get_server_url()  # type: ignore
    # strip username and password from server_url
    url = re.sub("https://.*@", "https://", url)
    return url


def launch_task(task: sky.Task, dry_run=False):
    if dry_run:
        print_tip("DRY RUN.")
        print_tip("Tip: Pipe this command to `| yq -P .` to get the pretty yaml config.\n")
        print(task.to_yaml_config())
        return

    request_id = sky.jobs.launch(task)  # type: ignore

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {bold(f'sky api logs {short_request_id}')}")
    print(f"- Or, visit: {bold(f'{dashboard_url()}/api/stream?request_id={short_request_id}')}")
    print("  - To sign in, use credentials from your ~/.skypilot/config.yaml file.")
    print(f"- To cancel the request, run: {bold(f'sky api cancel {short_request_id}')}")


def check_git_state(skip_check: bool = False) -> bool:
    """Check for uncommitted changes that won't be reflected in the cloud."""
    if skip_check:
        return True

    try:
        # Check for uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)

        if result.stdout.strip():
            print("❌ You have uncommitted changes that won't be reflected in the cloud job.")
            print("Options:")
            print("  - Commit: git add . && git commit -m 'your message'")
            print("  - Stash: git stash")
            print("  - Skip check: add --skip-git-check flag")
            return False

    except subprocess.CalledProcessError:
        print("⚠️  Could not check git status (not in a git repo?)")

    return True


def check_config_files(cmd_args: List[str]) -> bool:
    """Check that config files referenced in arguments actually exist."""
    config_files_to_check = []

    for task_arg in cmd_args:
        # Check for agent configuration
        if task_arg.startswith("agent="):
            agent_value = task_arg.split("=", 1)[1]
            config_files_to_check.append((task_arg, f"./configs/agent/{agent_value}.yaml"))

        # Check for trainer configuration
        elif task_arg.startswith("trainer="):
            trainer_value = task_arg.split("=", 1)[1]
            config_files_to_check.append((task_arg, f"./configs/trainer/{trainer_value}.yaml"))

        # Check for environment configuration
        elif task_arg.startswith("trainer.curriculum="):
            env_value = task_arg.split("=", 1)[1]
            config_files_to_check.append((task_arg, f"./configs/{env_value}.yaml"))

        # Check for evaluation configuration
        elif task_arg.startswith("trainer.sim="):
            sim_value = task_arg.split("=", 1)[1]
            config_files_to_check.append((task_arg, f"./configs/sim/{sim_value}.yaml"))

    missing_files = []
    for arg, config_path in config_files_to_check:
        if not Path(config_path).exists():
            missing_files.append((arg, config_path))

    if missing_files:
        print("❌ Config files not found:")
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
    task_args: List[str],
    git_ref: Optional[str] = None,
    commit_message: Optional[str] = None,
    timeout_minutes: Optional[int] = None,
    **kwargs,
) -> None:
    """Display a summary of the job that will be launched."""
    divider_length = 60
    divider = "=" * divider_length

    print(f"\n{divider}")
    print(bold("Job will be submitted with the following details:"))
    print(f"{divider}")

    print(f"Job Name: {job_name}")

    # Display any additional job details from kwargs
    for key, value in kwargs.items():
        if value is not None:
            # Convert snake_case to Title Case for display
            display_key = key.replace("_", " ").title()
            print(f"{display_key}: {value}")

    # Display timeout information with prominence
    if timeout_minutes:
        timeout_hours = timeout_minutes // 60
        timeout_mins = timeout_minutes % 60

        if timeout_hours > 0:
            if timeout_mins == 0:
                timeout_str = f"{timeout_hours}h"
            else:
                timeout_str = f"{timeout_hours}h {timeout_mins}m"
        else:
            timeout_str = f"{timeout_mins}m"

        print(bold(yellow(f"AUTO-TERMINATION: Job will terminate after {timeout_str}")))
    else:
        print(yellow("NO TIMEOUT SET: Job will run until completion"))

    # Display git information
    if git_ref:
        print(f"Git Reference: {git_ref}")
        if commit_message:
            first_line = commit_message.split("\n")[0]
            print(f"Commit Message: {yellow(first_line)}")

    print(f"{'-' * divider_length}")
    print(f"Command: {cmd}")

    # Display task arguments
    if task_args:
        print(yellow("Task Arguments:"))
        for i, arg in enumerate(task_args):
            print(f"  {i + 1}. {cyan(arg)}")

    print(f"\n{divider}")


def get_user_confirmation(
    dry_run: bool = False, skip_validation: bool = False, prompt: str = "Should we proceed?"
) -> bool:
    """Get user confirmation before proceeding with an action."""
    if dry_run:
        print(bold("DRY RUN - No action will be taken"))
        return False

    if not skip_validation:
        response = input(f"{prompt} (Y/n): ").strip().lower()
        if response not in ["", "y", "yes"]:
            print(yellow("Action cancelled by user."))
            return False

    return True


def validate_and_launch_task(
    task: sky.Task,
    cmd: str,
    task_args: List[str],
    git_ref: Optional[str] = None,
    timeout_hours: Optional[float] = None,
    copies: int = 1,
    dry_run: bool = False,
    skip_git_check: bool = False,
    use_color: bool = True,
) -> bool:
    """
    Simple validation and launch workflow focusing on common mistakes.

    Returns True if task was successfully launched (or would be in dry run mode).
    """
    # Set up colors
    use_colors(sys.stdout.isatty() and use_color)

    # Ensure we're in the repository root
    cd_repo_root()

    # Check the two things that commonly go wrong
    if not check_git_state(skip_git_check):
        return False

    if not check_config_files(task_args):
        return False

    # Launch the task(s)
    try:
        if copies == 1:
            launch_task(task, dry_run)
        else:
            for i in range(copies):
                copy_task = copy.deepcopy(task)
                copy_task.name = f"{task.name}_{i + 1}"
                copy_task = copy_task.update_envs({"METTA_RUN_ID": copy_task.name})
                copy_task.validate_name()
                print(f"\nLaunching copy {i + 1}/{copies}: {copy_task.name}")
                launch_task(copy_task, dry_run)

        if not dry_run:
            print(green(f"\n🚀 Successfully launched {copies} job{'s' if copies > 1 else ''}!"))

        return True

    except Exception as e:
        print(red(f"Failed to launch task: {e}"))
        return False
