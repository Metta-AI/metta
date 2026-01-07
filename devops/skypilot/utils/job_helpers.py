import logging
import re
import time
from io import StringIO, TextIOBase
from pathlib import Path
from typing import cast

import sky
import sky.exceptions
import sky.jobs
from sky.server.common import RequestId, get_server_url

logger = logging.getLogger(__name__)


SERVER_DOMAIN_SUFFIX = "softmax-research.net"


def get_devops_skypilot_dir() -> Path:
    return Path(__file__).parent.parent


def skypilot_sanity_check() -> None:
    server_url = get_server_url()
    if not server_url.endswith(SERVER_DOMAIN_SUFFIX):
        raise ValueError(f"Invalid SkyPilot server URL: {server_url}. Try `metta install skypilot --force` to fix.")


def get_jobs_controller_name() -> str:
    job_clusters = sky.get(sky.status(all_users=True, cluster_names=["sky-jobs-controller*"]))
    if len(job_clusters) == 0:
        raise ValueError("No job controller cluster found, is it running?")
    return job_clusters[0]["name"]


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
