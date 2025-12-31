"""
Helper functions for working with SkyPilot tasks.

These functions are used in launch.py and sandbox.py CLI scripts, so they often print to stdout.
"""

import copy
import netrc
import os
import re
from typing import Any, Optional

import sky
import sky.exceptions
import sky.jobs
from sky.server.common import get_server_url
from sky.utils.common_utils import check_cluster_name_is_valid

import gitta as git
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.text_styles import blue, bold, green, red, yellow


def patch_task(
    task: sky.Task,
    cpus: int | None,
    gpus: int | None,
    nodes: int | None,
    no_spot: bool = False,
) -> None:
    overrides: dict[str, Any] = {}
    if cpus:
        overrides["cpus"] = cpus
    if overrides:
        task.set_resources_override(overrides)
    if nodes:
        task.num_nodes = nodes

    new_resources_list: list[sky.Resources] = list(task.resources)

    if gpus:
        new_resources_list = []
        for res in list(task.resources):
            if not isinstance(res.accelerators, dict):
                # shouldn't happen with our current config
                raise Exception(f"Unexpected accelerator type: {res.accelerators}, {type(res.accelerators)}")

            patched_accelerators = copy.deepcopy(res.accelerators)
            patched_accelerators = {gpu_type: gpus for gpu_type in patched_accelerators.keys()}
            new_resources = res.copy(accelerators=patched_accelerators)
            new_resources_list.append(new_resources)

    if no_spot:
        new_resources_list = [res.copy(use_spot=False) for res in new_resources_list]

    if gpus or no_spot:
        task.set_resources(type(task.resources)(new_resources_list))


def validate_task_name(run_name: str) -> bool:
    """Validate that we will meet Sky's task naming requirements."""
    pattern = r"^[a-zA-Z]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$"
    valid = bool(re.match(pattern, run_name))

    try:
        check_cluster_name_is_valid(run_name)
    except sky.exceptions.InvalidClusterNameError:
        print(red(f"[VALIDATION] ❌ Invalid run name: '{run_name}'"))
        print("Sky task names must:")
        print("  - Start with a letter (not a number)")
        print("  - Contain only letters, numbers, dashes, underscores, or dots")
        print("  - End with a letter or number")
        print()

    return valid


def display_task_summary(
    task: sky.Task,
    commit_hash: str,
    git_ref: Optional[str] = None,
    skip_github: bool = False,
    copies: int = 1,
) -> None:
    """Display a summary of the task that will be launched.

    Args:
        skip_github: If True, skip GitHub API calls (for testing/CI)
    """
    divider_length = 60
    divider = blue("=" * divider_length)

    print(f"\n{divider}")
    print(bold(blue("Job details:")))
    print(f"{divider}")

    print(f"{bold('Name:')} {yellow(task.name)}")

    # Extract resource info from task
    if task.resources:
        resource = list(task.resources)[0]  # Get first resource option

        # GPU info
        if resource.accelerators:
            gpu_info: list[str] = []
            for gpu_type, count in resource.accelerators.items():
                gpu_info.append(f"{count}x {gpu_type}")
            print(f"{bold('GPUs:')} {yellow(', '.join(gpu_info))}")

        # CPU info
        if resource.cpus:
            print(f"{bold('CPUs:')} {yellow(str(resource.cpus))}")

        # Spot instance info
        spot_status = "Yes" if resource.use_spot else "No"
        print(f"{bold('Spot Instances:')} {yellow(spot_status)}")

    # Node count
    if task.num_nodes and task.num_nodes > 1:
        print(f"{bold('Nodes:')} {yellow(str(task.num_nodes))}")

    # Display any additional job details from kwargs
    if copies != 1:
        print(f"{bold('Copies:')} {yellow(copies)}")

    # Display timeout information with prominence
    timeout_hours = task.envs.get("MAX_RUNTIME_HOURS")
    if timeout_hours:
        timeout_mins = int(float(timeout_hours) * 60)
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

    # Only check GitHub if not skipped (avoids API calls in tests/CI)
    if not skip_github:
        try:
            pr_info = git.get_matched_pr(commit_hash, REPO_SLUG)
            if pr_info:
                pr_number, pr_title = pr_info
                first_line = pr_title.split("\n")[0]
                print(f"{bold('PR:')} {yellow(f'#{pr_number} - {first_line}')}")
            else:
                print(f"{bold('PR:')} {red('Not an open PR HEAD')}")
        except git.GitError:
            # GitHub API unavailable (rate limit, network error, etc.)
            # This is non-critical info, so we just skip it
            pass

    print(blue("-" * divider_length))

    cmd = f"{task.envs['METTA_MODULE_PATH']} (args: {task.envs['METTA_ARGS']})"
    print(f"\n{bold('Command:')} {yellow(cmd)}")

    run_id = task.envs.get("METTA_RUN_ID", task.name)
    dd_url = f"https://app.datadoghq.com/logs?query=metta_run_id:{run_id}"
    print(f"\n{bold('Datadog Logs:')} {yellow(dd_url)}")

    print(f"\n{divider}")


def set_task_secrets(task: sky.Task) -> None:
    """Write job secrets to task envs."""
    # Note: we can't mount these with `file_mounts` because of skypilot bug with service accounts.
    # Also, copying the entire `.netrc` is too much (it could contain other credentials).

    # Lazy import to avoid loading wandb at CLI startup
    import wandb

    wandb_password = netrc.netrc(os.path.expanduser("~/.netrc")).hosts["api.wandb.ai"][2]
    if not wandb_password:
        raise ValueError("Failed to get wandb password, run 'metta install' to fix")

    # Lazy import - app_backend is optional and not available in CI
    try:
        from metta.app_backend.clients.base_client import get_machine_token

        observatory_token = get_machine_token(PROD_STATS_SERVER_URI)
    except ImportError:
        observatory_token = None

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


def launch_task(task: sky.Task) -> str:
    request_id = sky.jobs.launch(task)

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {yellow(f'sky api logs {short_request_id}')}")
    dashboard_url = get_server_url() + "/dashboard/jobs"
    print(f"- Or, visit: {yellow(dashboard_url)}")

    from devops.skypilot.utils.job_helpers import get_job_id_from_request_id

    job_id = get_job_id_from_request_id(request_id, wait_seconds=5.0)
    if job_id:
        print(f"Job ID: {job_id}")
    else:
        print("Job ID: pending")

    return request_id
