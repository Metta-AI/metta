import logging
import netrc
import os
import re
import time
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Optional, cast

import sky
import sky.exceptions
import sky.jobs
from sky.server.common import RequestId, get_server_url

import gitta as git
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.text_styles import blue, bold, green, red, yellow

logger = logging.getLogger(__name__)


def get_devops_skypilot_dir() -> Path:
    return Path(__file__).parent.parent


def get_jobs_controller_name() -> str:
    job_clusters = sky.get(sky.status(all_users=True, cluster_names=["sky-jobs-controller*"]))
    if len(job_clusters) == 0:
        raise ValueError("No job controller cluster found, is it running?")
    return job_clusters[0]["name"]


def launch_task(task: sky.Task) -> str:
    request_id = sky.jobs.launch(task)

    print(green(f"Submitted sky.jobs.launch request: {request_id}"))

    short_request_id = request_id.split("-")[0]

    print(f"- Check logs with: {yellow(f'sky api logs {short_request_id}')}")
    dashboard_url = get_server_url() + "/dashboard/jobs"
    print(f"- Or, visit: {yellow(dashboard_url)}")

    return request_id


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

        observatory_token = get_machine_token("https://api.observatory.softmax-research.net")
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
    """Get job ID from a request ID."""
    time.sleep(wait_seconds)  # Wait for job to be registered

    try:
        job_id, _ = sky.get(RequestId(request_id))
        return str(job_id) if job_id is not None else None
    except Exception:
        return None


def check_job_statuses(job_ids: list[int]) -> dict[int, dict[str, str]]:
    """Check the status of multiple jobs using the SDK."""
    if not job_ids:
        return {}

    job_data = {}

    try:
        # Get job queue filtered to only the requested job IDs (more efficient than fetching all jobs)
        job_records = sky.get(sky.jobs.queue(refresh=True, all_users=True, job_ids=job_ids))

        # Create a mapping for quick lookup
        jobs_map = {job["job_id"]: job for job in job_records}

        for job_id in job_ids:
            if job_id in jobs_map:
                job = jobs_map[job_id]

                # Calculate time ago
                submitted_timestamp = job.get("submitted_at") or time.time()
                time_diff = time.time() - submitted_timestamp

                if time_diff < 60:
                    time_ago = f"{int(time_diff)} secs ago"
                elif time_diff < 3600:
                    time_ago = f"{int(time_diff / 60)} mins ago"
                else:
                    time_ago = f"{int(time_diff / 3600)} hours ago"

                # Format duration
                duration = job.get("job_duration") or 0
                if duration:
                    duration_str = f"{int(duration)}s" if duration < 60 else f"{int(duration / 60)}m"
                else:
                    duration_str = ""

                job_data[job_id] = {
                    "status": str(job["status"]).split(".")[-1],  # Extract status name
                    "name": job.get("job_name", ""),
                    "submitted": time_ago,
                    "duration": duration_str,
                    "raw_line": "",  # SDK doesn't provide raw line format
                }
            else:
                job_data[job_id] = {"status": "UNKNOWN", "name": "", "submitted": "", "duration": "", "raw_line": ""}

    except sky.exceptions.ClusterNotUpError as e:
        # Jobs controller not up - treat as temporary API issue
        logger.warning(f"SkyPilot API error: {e}")
        for job_id in job_ids:
            job_data[job_id] = {"status": "ERROR", "raw_line": "", "error": "Jobs controller not up"}
    except Exception as e:
        # API/network errors - will timeout after error_timeout_s if persistent
        logger.warning(f"SkyPilot API error: {e}")
        for job_id in job_ids:
            job_data[job_id] = {"status": "ERROR", "raw_line": "", "error": str(e)}

    return job_data


def tail_job_log(job_id: str, lines: int = 100) -> str | None:
    """Get the tail of job logs using the SDK. Always returns last few lines."""
    try:
        # First check if the job exists and get its status
        job_status = sky.get(sky.job_status(get_jobs_controller_name(), job_ids=[int(job_id)]))

        if job_status.get(int(job_id)) is None:
            return f"Error: Job {job_id} not found"

        # Use a StringIO to capture output
        output = StringIO()

        try:
            # Try to get logs without following
            sky.jobs.tail_logs(job_id=int(job_id), follow=False, tail=lines, output_stream=cast(TextIOBase, output))

            result = output.getvalue()
            if result:
                return result
            else:
                # No logs yet, job might be provisioning
                return f"No logs available yet for job {job_id} (may still be provisioning)"

        except sky.exceptions.ClusterNotUpError:
            return "Error: Jobs controller is not up"
        except Exception as e:
            # If there's an error getting logs, provide context
            error_msg = str(e)
            if "still running" in error_msg.lower():
                # Try one more time with preload_content=False if available
                # Otherwise just indicate it's running
                return f"Job {job_id} is currently running (logs may be incomplete)"
            else:
                return f"Error retrieving logs for job {job_id}: {error_msg}"

    except ValueError:
        return f"Error: Invalid job ID format: {job_id}"
    except Exception as e:
        return f"Error: {str(e)}"
