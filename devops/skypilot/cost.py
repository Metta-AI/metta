#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "skypilot",
#     "typer",
#     "rich",
# ]
# ///
"""
CLI tool to calculate costs for SkyPilot jobs over a specified time period.
"""

import logging
import re
import sys
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, Optional

import sky
import sky.clouds
import sky.exceptions
import sky.jobs
import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    help="Track SkyPilot job costs over time",
    add_completion=False,
)

# SkyPilot job type.
Job = dict[str, Any]

# Return types for helper functions
InstanceInfo = namedtuple("InstanceInfo", ["instance_type", "num_nodes"])
GpuOption = namedtuple("GpuOption", ["gpu_type", "num_gpus", "num_nodes"])
CostInfo = namedtuple("CostInfo", ["cost", "status"])


def _instance_info_from_cluster_resources(cluster_resources: str, /) -> InstanceInfo | None:
    """
    Parses instance type and node count from a cluster resources string.

    Returns None if the string is empty or cannot be parsed.

    Example:

        >>> _instance_info_from_cluster_resources("")
        None

        >>> _instance_info_from_cluster_resources("1x(gpus=L4:4, g6.12xlarge, ...)")
        InstanceInfo(instance_type="g6.12xlarge", num_nodes=1)

    """

    if cluster_resources == "" or cluster_resources == "-":
        return None

    # Extract node count - pattern like "2x(" or "1x("
    nodes_match = re.search(r"(\d+)x\(", cluster_resources)
    if not nodes_match:
        return None
    num_nodes = int(nodes_match.group(1))

    # Extract instance type - look for AWS instance patterns like "g6.12xlarge" or "p4d.24xlarge"
    instance_match = re.search(r"\b([a-z][a-z0-9-]*\.\w+)\b", cluster_resources)
    if not instance_match:
        return None
    instance_type = instance_match.group(1)

    return InstanceInfo(instance_type=instance_type, num_nodes=num_nodes)


def _gpu_options_from_job_resources(
    job_resources: str,
    /,
) -> list[GpuOption] | None:
    """
    Parses GPU type, GPU count, and node count from a job resources string.

    Returns None if string is empty or cannot be parsed.

    Example:

        >>> _gpu_options_from_job_resources("")
        None

        >>> _gpu_options_from_job_resources("2x[L4:2, A10G:1]")
        [
            GpuInfo(gpu_type='L4', num_gpus=2, num_nodes=2),
            GpuInfo(gpu_type='A10G', num_gpus=1, num_nodes=2),
        ]

    """

    if job_resources == "":
        return None

    # Extract node count - pattern like "2x[" or "1x["
    nodes_match = re.search(r"(\d+)x\[", job_resources)
    if nodes_match is None:
        return None
    num_nodes = int(nodes_match.group(1))

    # Extract all GPU type and count pairs - patterns like "L4:1", "A10G:2", "H100:8"
    gpu_matches = re.findall(r"(\w+):(\d+)", job_resources)
    if not gpu_matches:
        return None

    return [
        GpuOption(gpu_type=gpu_type, num_gpus=int(num_gpus_str), num_nodes=num_nodes)
        for gpu_type, num_gpus_str in gpu_matches
    ]


def _instance_type_from_gpu_option(*, gpu_type: str, num_gpus: int) -> str | None:
    """
    Map GPU type and count to a AWS instance type.
    """

    # Static mapping from GPU type and count to AWS instance.
    # This only changes once or twice a year and can be updated manually.
    # Source: https://aws.amazon.com/ec2/instance-types/
    GPU_INSTANCE_MAP = {
        "T4": {
            1: "g4dn.xlarge",
            4: "g4dn.12xlarge",
            8: "g4dn.metal",
        },
        "A10G": {
            1: "g5.xlarge",
            4: "g5.12xlarge",
            8: "g5.48xlarge",
        },
        "L4": {
            1: "g6.xlarge",
            4: "g6.12xlarge",
            8: "g6.48xlarge",
        },
        "V100": {
            1: "p3.2xlarge",
            4: "p3.8xlarge",
            8: "p3.16xlarge",
        },
        "A100": {
            8: "p4d.24xlarge",
        },
        "H100": {
            8: "p5.48xlarge",
        },
    }

    if gpu_type in GPU_INSTANCE_MAP and num_gpus in GPU_INSTANCE_MAP[gpu_type]:
        return GPU_INSTANCE_MAP[gpu_type][num_gpus]
    else:
        return None


def _get_instance_hourly_cost(
    *,
    instance_type: str,
    region: str,
    use_spot: bool,
) -> float | None:
    try:
        cloud = sky.clouds.AWS()
        return cloud.instance_type_to_hourly_cost(
            instance_type,
            use_spot=use_spot,
            region=region,
        )
    except Exception as e:
        logger.debug(f"Could not get cost for {instance_type}: {e}")
        return None


def _select_cheapest_gpu_option(
    *,
    gpu_options: list[GpuOption],
    region: str,
    use_spot: bool,
) -> InstanceInfo | None:
    """
    Selects the cheapest GPU option from multiple alternatives.

    When a job's resources field contains multiple GPU types (e.g., '1x[A10G:4, L4:4]'),
    it means the job accepts EITHER option. SkyPilot picks the cheapest one at runtime.
    This function replicates that selection to get accurate cost estimates.
    """
    cheapest_instance_info = None
    cheapest_cost = float("inf")

    for gpu_info in gpu_options:
        instance_type = _instance_type_from_gpu_option(gpu_type=gpu_info.gpu_type, num_gpus=gpu_info.num_gpus)
        if not instance_type:
            logger.debug(f"Could not map {gpu_info.gpu_type}:{gpu_info.num_gpus} to instance type")
            continue

        hourly_cost = _get_instance_hourly_cost(instance_type=instance_type, region=region, use_spot=use_spot)
        if hourly_cost is None:
            logger.debug(f"Could not get cost for {instance_type}")
            continue

        if hourly_cost < cheapest_cost:
            cheapest_cost = hourly_cost
            cheapest_instance_info = InstanceInfo(instance_type=instance_type, num_nodes=gpu_info.num_nodes)

    return cheapest_instance_info


def _duration_hours_from_job(job: Job, /) -> float | None:
    if job["job_duration"] is None:
        return None
    return job["job_duration"] / 3600.0


def _region_from_job(job: Job, /) -> str:
    # Use the job's `region` field if available.
    if job["region"] != "-":
        return job["region"]

    # Use the region in the job's `user_yaml` if available.
    if user_yaml := job["user_yaml"]:
        # Look for newer "region: <name>" pattern.
        if region_match := re.search(r"region:\s*([a-z0-9-]+)", user_yaml, re.IGNORECASE):
            return region_match.group(1)

        # Try older "infra: <cloud>/<region>" format.
        if infra_match := re.search(r"infra:\s*([a-z]+)/([a-z0-9-]+)", user_yaml, re.IGNORECASE):
            return infra_match.group(2)

    # Fallback to default region.
    logger.warning("Could not determine region from job, defaulting to us-east-1")
    return "us-east-1"


def _use_spot_from_job(job: Job, /) -> bool:
    # Use the job's `user_yaml` if available.
    if user_yaml := job["user_yaml"]:
        if use_spot_match := re.search(r"use_spot:\s*(true|false)", user_yaml):
            return use_spot_match.group(1) == "true"

    # Use the info in job `resources` field if available.
    if "Spot" in job["resources"]:
        return True
    if "On-Demand" in job["resources"]:
        return False

    # Fallback to default. Better to overestimate cost than to underestimate it.
    logger.warning("Could not determine use_spot from job, defaulting to False")
    return False


def _cost_info_from_job(job: Job, /) -> CostInfo:
    """
    Calculate cost for a single job.

    Args:
        job: Job record from SkyPilot

    Returns:
        JobInfo(cost, status) where cost is in USD or None, and status is an error message or "OK"
    """
    region = _region_from_job(job)
    use_spot = _use_spot_from_job(job)
    duration_hours = _duration_hours_from_job(job)
    if duration_hours is None:
        return CostInfo(cost=None, status="No duration")

    # If we have instance type from `cluster_resources`, use that first as it's more accurate.
    if _instance_info := _instance_info_from_cluster_resources(job["cluster_resources"]):
        instance_info = _instance_info

    else:
        # Otherwise, parse GPU options requested in `job_resources`.
        gpu_options = _gpu_options_from_job_resources(job["resources"])
        if gpu_options is None:
            return CostInfo(cost=None, status="Could not parse GPU info from job resources")

        # When multiple GPU options were requested, select the cheapest option
        #  to replicate the behavior of SkyPilot.
        instance_info = _select_cheapest_gpu_option(gpu_options=gpu_options, region=region, use_spot=use_spot)
        if instance_info is None:
            return CostInfo(cost=None, status="Could not determine instance type for any GPU option")

    hourly_cost = _get_instance_hourly_cost(instance_type=instance_info.instance_type, region=region, use_spot=use_spot)
    if hourly_cost is None:
        return CostInfo(cost=None, status=f"Cost unavailable for {instance_info.instance_type}")

    total_cost = hourly_cost * instance_info.num_nodes * duration_hours
    return CostInfo(cost=total_cost, status="OK")


def _format_duration(*, seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    else:
        hours = seconds / 3600.0
        return f"{hours:.1f}h"


def _format_timestamp(*, timestamp: float | None) -> str:
    """Format timestamp to human-readable date."""
    if not timestamp:
        return "N/A"

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M")


_DEFAULT_DAYS = 7


@app.command()
def main(
    job_id: Optional[int] = typer.Option(None, "--job-id", help="Show costs for a specific job ID"),
    days: Optional[int] = typer.Option(None, "--days", help=f"Number of days to look back (default: {_DEFAULT_DAYS})"),
    me: bool = typer.Option(False, "--me", help="Show only your jobs (default unless --everyone or --user specified)"),
    everyone: bool = typer.Option(False, "--everyone", help="Show all users' jobs"),
    user: Optional[str] = typer.Option(None, "--user", help="Show jobs for a specific user (email or username)"),
):
    """
    Track SkyPilot job costs over a specified time period.

    By default, shows only YOUR jobs. Use --everyone to see all users' jobs
    or --user to see jobs for a specific user.

    Note: Costs are estimated using CURRENT pricing rates from cloud providers.
    If pricing has changed since jobs ran, estimates may differ from actual charges.

    Examples:

        # Show YOUR costs for last 7 days (default)
        python devops/skypilot/cost.py

        # Show YOUR costs for last 30 days
        python devops/skypilot/cost.py --days 30

        # Show EVERYONE's costs
        python devops/skypilot/cost.py --everyone --days 7

        # Show cost for a specific job
        python devops/skypilot/cost.py --job-id 123

        # Show jobs for a specific user
        python devops/skypilot/cost.py --user yatharth@softmax.com
    """

    try:
        ################################################################################
        # Fetch and filter jobs.
        ################################################################################

        if job_id:
            # If `--job-id` is specified, shouldn't provide any other filters.
            if me or everyone or user or days:
                console.print("[red]Error: Cannot specify --job-id with other filters[/red]")
                sys.exit(1)

            # Fetch the job.
            console.print("[dim]Querying SkyPilot jobs...[/dim]")
            jobs = sky.get(sky.jobs.queue(refresh=False, all_users=True, job_ids=[job_id]))

            # Handle job not found.
            if len(jobs) == 0:
                console.print(f"[red]No job found with ID {job_id}[/red]")
                sys.exit(1)

        else:
            # Validate user filter options — ensure mutual exclusivity.
            if sum([me, everyone, user is not None]) > 1:
                console.print("[red]Error: Can only specify one of --me, --everyone, or --user[/red]")
                sys.exit(1)

            # Default to --me if none specified.
            if not everyone and not user:
                me = True

            # Fetch jobs and filter by user.
            console.print("[dim]Querying SkyPilot jobs...[/dim]")
            if me:
                jobs = sky.get(sky.jobs.queue(refresh=False, all_users=False))
            elif everyone:
                jobs = sky.get(sky.jobs.queue(refresh=False, all_users=True))
            elif user:
                jobs = sky.get(sky.jobs.queue(refresh=False, all_users=True))
                jobs = [job for job in jobs if job["user_name"].lower().find(user.lower()) >= 0]
            else:
                raise AssertionError("Unreachable")

            # Default to certain number of days if `--days` is not specified.
            if days is None:
                days = _DEFAULT_DAYS

            # Filter jobs by time.
            _cutoff_time = time.time() - (days * 24 * 3600)
            jobs = [
                job
                for job in jobs
                # Include any jobs still running...
                if (end_at := job["end_at"]) is None
                # ... or those completed in the last `days` days.
                or end_at >= _cutoff_time
            ]
            if len(jobs) == 0:
                console.print(f"[yellow]No jobs found in the last {days} days[/yellow]")
                return

        ################################################################################
        # Calculate costs for each job.
        ################################################################################

        console.print("[dim]Calculating costs...[/dim]")

        jobs_with_cost_info = []
        total_cost = 0.0
        failed_count = 0
        for job in jobs:
            cost_info = _cost_info_from_job(job)

            if cost_info.cost is not None:
                total_cost += cost_info.cost
            else:
                failed_count += 1

            jobs_with_cost_info.append(
                {
                    "job": job,
                    "cost": cost_info.cost,
                    "status": cost_info.status,
                }
            )

        # Sort by end time (most recent first), with running jobs at the top
        jobs_with_cost_info.sort(
            key=lambda x: x["job"]["end_at"] or time.time(),
            reverse=True,
        )

        ################################################################################
        # Print jobs table.
        ################################################################################

        title_parts = []
        if job_id:
            title_parts.append(f"Job {job_id}")
        else:
            title_parts.append(f"Jobs over last {days} days")

        if me:
            title_parts.append("(you)")
        elif user:
            title_parts.append(f"(user: {user})")
        elif everyone:
            title_parts.append("(all users)")

        table = Table(title=" ".join(title_parts))
        table.add_column("Job ID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("End Time", justify="left", style="dim")

        # Add user column only when showing everyone's jobs
        show_user_column = everyone
        if show_user_column:
            table.add_column("User", justify="left", style="dim")

        for item in jobs_with_cost_info:
            job = item["job"]

            # Format job ID and name.
            job_id_str = str(job["job_id"])
            job_name = job["job_name"]

            # Format duration.
            duration = job["job_duration"]
            duration_str = _format_duration(seconds=duration) if duration else "N/A"

            # Format status.
            status = job["status"]
            if status:
                status_str = str(status).split(".")[-1]
            else:
                status_str = "UNKNOWN"

            # Color code status.
            if "RUNNING" in status_str.upper():
                status_display = f"[blue]{status_str}[/blue]"
            elif "SUCCEEDED" in status_str.upper():
                status_display = f"[green]{status_str}[/green]"
            elif "FAILED" in status_str.upper() or "CANCELLED" in status_str.upper():
                status_display = f"[red]{status_str}[/red]"
            else:
                status_display = status_str

            # Format cost.
            cost = item["cost"]
            if cost is not None:
                cost_str = f"${cost:.2f}"
            else:
                cost_str = f"[dim]N/A ({item['status']})[/dim]"

            # Format end time.
            end_time = _format_timestamp(timestamp=job["end_at"])
            if not job["end_at"]:
                end_time = "[blue]Running[/blue]"

            # Compose the row.
            row_data = [
                job_id_str,
                job_name,
                duration_str,
                status_display,
                cost_str,
                end_time,
            ]

            # Add optional user column.
            if show_user_column:
                # Format user.
                user_name = job["user_name"]
                if "@" in user_name:
                    user_name = user_name.split("@")[0]
                row_data.append(user_name)

            table.add_row(*row_data)

        console.print()
        console.print(table)
        console.print()

        ################################################################################
        # Print user cost summary table (only in --everyone mode)
        ################################################################################

        if everyone:
            user_costs = defaultdict(float)
            for item in jobs_with_cost_info:
                if item["cost"] is not None:
                    job = item["job"]
                    user_name = job["user_name"]
                    # Extract username from email if present
                    if "@" in user_name:
                        user_name = user_name.split("@")[0]
                    user_costs[user_name] += item["cost"]

            # Sort by cost descending
            sorted_users = sorted(user_costs.items(), key=lambda x: x[1], reverse=True)

            # Create summary table
            summary_table = Table(title=f"Total Cost per User (last {days} days)")
            summary_table.add_column("User", justify="left", style="cyan")
            summary_table.add_column("Total Cost", justify="right", style="yellow")

            for user, cost in sorted_users:
                summary_table.add_row(user, f"${cost:,.2f}")

            console.print()
            console.print(summary_table)
            console.print()

        ################################################################################
        # Print final summary.
        ################################################################################

        console.print(f"[bold]Total Cost: [yellow]${total_cost:,.2f}[/yellow][/bold]")
        console.print(f"[dim]Jobs analyzed: {len(jobs_with_cost_info):,}[/dim]")

        if failed_count > 0:
            console.print(f"[dim yellow]Note: Could not calculate cost for {failed_count:,} job(s)[/dim yellow]")

        console.print()
        console.print("[dim]💡 Costs are estimated using CURRENT cloud pricing rates.[/dim]")
        console.print("[dim]   If pricing changed since jobs ran, estimates may differ from actual bills.[/dim]")

    except sky.exceptions.ClusterNotUpError:
        console.print("[red]Error: Jobs controller is not running[/red]")
        console.print("[dim]Try starting it with: sky start sky-jobs-controller[/dim]")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Failed to get cost information")
        sys.exit(1)


if __name__ == "__main__":
    app()
