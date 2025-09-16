#!/usr/bin/env uv run
"""Live run monitor with rich terminal display.

This utility provides a real-time, color-coded view of runs with automatic refresh.

Features:
- Color-coded status: Completed (blue), In Training (green), Pending (gray),
  Training Done No Eval (orange), Failed (red)
- Live progress updates in Gsteps (billions of steps)
- Cost tracking in USD format
- Auto-refresh every 30 seconds (configurable)
- In-place updates (no scrolling output)
- Runs sorted by most recent first
- Configurable fetch and display limits to prevent excessive WandB queries
- Default: fetch 50 runs, display 10 most recent
- Configurable summary fields from wandb run.summary with custom display names

Usage:
    ./metta/util/live_run_monitor.py --group my_group_name
    ./metta/util/live_run_monitor.py --name-filter "axel.*"
    ./metta/util/live_run_monitor.py --group my_group --name-filter "experiment.*"
    ./metta/util/live_run_monitor.py --refresh 15 --entity myteam --project myproject
    ./metta/util/live_run_monitor.py --fetch-limit 100 --display-limit 20
    ./metta/util/live_run_monitor.py  # Monitor last 10 runs (fetch 50, display 10)
    ./metta/util/live_run_monitor.py --summary-fields "env_agent/heart.get:Score,env_agent/food.get:Food"
    ./metta/util/live_run_monitor.py --summary-fields "train/loss:Loss,eval/accuracy:Acc,cost:Cost"
"""

import argparse
import logging
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import wandb
from dateutil import parser
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.text import Text as RichText
from sparklines import sparklines

from metta.sweep.models import JobStatus, RunInfo

logger = logging.getLogger(__name__)

# Display limit for runs
DISPLAY_LIMIT = 10


class ApiRequestTracker:
    """Track API requests and calculate request rates."""

    def __init__(self):
        self.total_requests = 0
        self.request_times: List[float] = []
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_request(self, count: int = 1):
        """Record an API request."""
        with self._lock:
            self.total_requests += count
            current_time = time.time()
            self.request_times.append(current_time)
            # Keep only requests from the last 60 seconds for rate calculation
            cutoff = current_time - 60
            self.request_times = [t for t in self.request_times if t > cutoff]

    def get_stats(self) -> Tuple[int, float]:
        """Get total requests and requests per minute rate.

        Returns:
            Tuple of (total_requests, requests_per_minute)
        """
        with self._lock:
            # Calculate rate based on requests in the last minute
            current_time = time.time()
            elapsed = current_time - self.start_time

            if elapsed < 5:
                # Not enough data to calculate rate
                rate = 0.0
            elif elapsed < 60:
                # If less than a minute has elapsed, extrapolate conservatively
                # Only extrapolate if we have at least 5 seconds of data
                rate = (len(self.request_times) / elapsed) * 60
            else:
                # Use actual count from last minute
                rate = len(self.request_times)

            return self.total_requests, rate


class MetricHistoryCache:
    """Cache for metric history data with background fetching.

    This class manages cached metric data and coordinates background fetching
    to avoid blocking the main rendering thread.
    """

    def __init__(self, entity: str, project: str, api_tracker: Optional[ApiRequestTracker] = None):
        """Initialize the metric history cache.

        Args:
            entity: WandB entity name
            project: WandB project name
            api_tracker: Optional API request tracker
        """
        self.entity = entity
        self.project = project
        self.api_tracker = api_tracker
        self.cache: Dict[str, Dict[str, List[float]]] = {}
        self.cache_lock = threading.Lock()
        self.fetch_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self._last_fetch_time: Dict[str, float] = {}
        # Track which run IDs are currently being fetched
        self._fetching: set[str] = set()
        # Minimum time between fetches for the same run (in seconds)
        self.min_fetch_interval = 10.0

    def start(self):
        """Start the background worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            logger.debug("Started metric history background worker")

    def stop(self):
        """Stop the background worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            # Add a sentinel value to unblock the queue
            self.fetch_queue.put(None)
            self.worker_thread.join(timeout=1.0)
            logger.debug("Stopped metric history background worker")

    def request_metrics(self, run_id: str, metric_keys: List[str]):
        """Request metrics to be fetched in the background.

        Args:
            run_id: The wandb run ID
            metric_keys: List of metric keys to fetch
        """
        # Check if we've fetched this run recently
        current_time = time.time()
        last_fetch = self._last_fetch_time.get(run_id, 0)
        if current_time - last_fetch < self.min_fetch_interval:
            return  # Skip if fetched recently

        try:
            self.fetch_queue.put_nowait((run_id, metric_keys))
            with self.cache_lock:
                self._fetching.add(run_id)
        except queue.Full:
            # Queue is full, skip this request
            logger.debug(f"Metric fetch queue full, skipping request for {run_id}")

    def get_metrics(self, run_id: str) -> Dict[str, List[float]]:
        """Get cached metrics for a run (non-blocking).

        Args:
            run_id: The wandb run ID

        Returns:
            Dictionary of metric keys to value lists, or empty dict if not cached
        """
        with self.cache_lock:
            return self.cache.get(run_id, {}).copy()

    def is_fetching(self, run_id: str) -> bool:
        """Check if metrics for a run are currently being fetched.

        Args:
            run_id: The wandb run ID

        Returns:
            True if metrics are being fetched, False otherwise
        """
        with self.cache_lock:
            return run_id in self._fetching

    def _worker(self):
        """Background worker that fetches metric history."""
        while not self.stop_event.is_set():
            try:
                # Get next fetch request with timeout
                request = self.fetch_queue.get(timeout=1.0)
                if request is None:  # Sentinel value
                    break

                run_id, metric_keys = request

                # Fetch the metrics
                try:
                    metrics = self._fetch_metrics_internal(run_id, metric_keys)

                    # Update cache
                    with self.cache_lock:
                        if run_id not in self.cache:
                            self.cache[run_id] = {}
                        self.cache[run_id].update(metrics)
                        # Remove from fetching set
                        self._fetching.discard(run_id)

                    # Update last fetch time
                    self._last_fetch_time[run_id] = time.time()

                    logger.debug(f"Fetched metrics for {run_id}: {list(metrics.keys())}")

                except Exception as e:
                    logger.debug(f"Error fetching metrics for {run_id}: {e}")
                    # Remove from fetching set even on error
                    with self.cache_lock:
                        self._fetching.discard(run_id)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in metric history worker: {e}")

    def _fetch_metrics_internal(self, run_id: str, metric_keys: List[str]) -> Dict[str, List[float]]:
        """Internal method to fetch metrics from WandB.

        Args:
            run_id: The wandb run ID
            metric_keys: List of metric keys to fetch

        Returns:
            Dictionary mapping metric keys to lists of values
        """
        try:
            api = wandb.Api()
            # Record API request for getting the run
            if self.api_tracker:
                self.api_tracker.record_request(1)
            run = api.run(f"{self.entity}/{self.project}/{run_id}")

            # For incremental fetching, we could track the last step fetched
            # and use run.history(min_step=last_step) but for now we'll
            # fetch the latest 100 samples
            # Record API request for fetching history
            if self.api_tracker:
                self.api_tracker.record_request(1)
            history = run.history(keys=metric_keys, samples=100)

            result = {}
            for key in metric_keys:
                if key in history.columns:
                    values = history[key].dropna().tolist()
                    if values:
                        result[key] = values
                    else:
                        result[key] = []
                else:
                    result[key] = []

            return result
        except Exception as e:
            logger.debug(f"Error fetching history for run {run_id}: {e}")
            return {key: [] for key in metric_keys}


def format_idle_time(last_updated: Optional[datetime], created_at: Optional[datetime] = None) -> str:
    """Format the time since last update in the most appropriate unit.

    Args:
        last_updated: The last update timestamp
        created_at: The creation timestamp (to cap idle time to job age)

    Returns:
        Formatted idle time string
    """
    if not last_updated:
        return "N/A"

    try:
        # Convert to datetime if it's a string
        if isinstance(last_updated, str):
            last_updated = parser.parse(last_updated)

        if created_at and isinstance(created_at, str):
            created_at = parser.parse(created_at)

        # Normalize timezone handling - always work in UTC
        current_time = datetime.now(timezone.utc)

        # Convert timestamps to UTC if they have timezone info, or assume UTC if naive
        if last_updated.tzinfo is None:
            # Assume naive datetime is UTC
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        elif last_updated.tzinfo != timezone.utc:
            # Convert to UTC
            last_updated = last_updated.astimezone(timezone.utc)

        # Check for epoch/default timestamps (1970-01-01) which indicate uninitialized data
        epoch_cutoff = datetime(1980, 1, 1, tzinfo=timezone.utc)  # Anything before 1980 is suspicious
        if last_updated < epoch_cutoff:
            logger.debug(f"Timestamp too old (likely uninitialized): last_updated={last_updated}")
            return "N/A"

        if created_at:
            if created_at.tzinfo is None:
                # Assume naive datetime is UTC
                created_at = created_at.replace(tzinfo=timezone.utc)
            elif created_at.tzinfo != timezone.utc:
                # Convert to UTC
                created_at = created_at.astimezone(timezone.utc)

            # Check if created_at is also suspiciously old
            if created_at < epoch_cutoff:
                logger.debug(f"Created timestamp too old (likely uninitialized): created_at={created_at}")
                return "N/A"

        # Calculate time difference
        delta = current_time - last_updated
        total_seconds = delta.total_seconds()

        # Sanity check - if the time is negative (future timestamp), something's wrong
        if total_seconds < 0:
            logger.debug(f"Negative idle time: last_updated={last_updated}, current={current_time}")
            return "N/A"

        # Cap idle time to job age if created_at is provided
        if created_at:
            job_age_delta = current_time - created_at
            job_age_seconds = job_age_delta.total_seconds()

            # If job age is negative or zero, something is wrong with timestamps
            if job_age_seconds <= 0:
                logger.debug(f"Invalid job age: created_at={created_at}, current={current_time}")
                return "N/A"

            # Idle time cannot be more than the job age
            total_seconds = min(total_seconds, job_age_seconds)

        # Additional sanity check for unreasonable values
        if total_seconds > 31536000:  # More than 365 days
            # Log this unusual case for debugging
            logger.debug(
                f"Idle time > 365 days: last_updated={last_updated}, created_at={created_at}, seconds={total_seconds}"
            )
            return "365d+"

        # Choose appropriate unit
        if total_seconds < 60:
            return f"{int(total_seconds)}s"
        elif total_seconds < 3600:  # Less than 1 hour
            minutes = int(total_seconds / 60)
            return f"{minutes}m"
        elif total_seconds < 86400:  # Less than 1 day
            hours = int(total_seconds / 3600)
            return f"{hours}h"
        else:  # Days
            days = int(total_seconds / 86400)
            if days == 1:
                return "1d"
            else:
                return f"{days}d"
    except Exception as e:
        logger.debug(f"Error in format_idle_time: {e}, last_updated={last_updated}, created_at={created_at}")
        return "N/A"


def _get_status_color(status: JobStatus) -> str:
    """Get color for run status."""
    if status == JobStatus.COMPLETED or status == JobStatus.EVAL_DONE_NOT_COMPLETED:
        return "bright_blue"
    elif status == JobStatus.IN_TRAINING:
        return "bright_green"
    elif status == JobStatus.PENDING:
        return "bright_black"
    elif status == JobStatus.TRAINING_DONE_NO_EVAL:
        return "green"
    elif status == JobStatus.IN_EVAL:
        return "bright_cyan"
    elif status == JobStatus.FAILED:
        return "bright_red"
    elif status == JobStatus.STALE:
        return "bright_black"
    else:
        return "white"


def generate_sparkline(values: List[float], width: int = 20, is_loading: bool = False) -> str:
    """Generate ASCII sparkline from a list of values using the sparklines library.

    Args:
        values: List of numeric values to plot
        width: Target width of the sparkline in characters
        is_loading: Whether the data is currently being fetched

    Returns:
        ASCII sparkline string, loading indicator, or dashed line
    """
    # Show loading indicator if data is being fetched
    if is_loading and (not values or len(values) < 2):
        # Create an animated loading pattern
        loading_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        # Use current time to cycle through loading chars
        idx = int(time.time() * 2) % len(loading_chars)
        # Just show the spinner followed by dashes
        return loading_chars[idx] + " " + "─" * (width - 2)

    # Show dashed line if no data available
    if not values or len(values) < 2:
        return "─" * width

    try:
        # Resample values if needed to fit width
        if len(values) > width:
            # Simple downsampling - take evenly spaced points
            indices = [int(i * (len(values) - 1) / (width - 1)) for i in range(width)]
            values = [values[i] for i in indices]

        # Generate sparkline using the library
        result = sparklines(values)
        if result and len(result) > 0:
            # Pad or truncate to exact width
            sparkline = result[0]
            if len(sparkline) < width:
                sparkline += " " * (width - len(sparkline))
            elif len(sparkline) > width:
                sparkline = sparkline[:width]
            return sparkline
        else:
            return "─" * width
    except ImportError:
        # Fallback if sparklines package is not available
        return "─" * width
    except Exception as e:
        logger.debug(f"Error generating sparkline: {e}")
        return "─" * width


def make_rich_monitor_table(
    runs: List[RunInfo],
    summary_fields: List[tuple[str, str]],
    sparkline_fields: Optional[List[tuple[str, str]]] = None,
    metric_cache: Optional[MetricHistoryCache] = None,
) -> Table:
    """Create rich table for run monitoring.

    Args:
        runs: List of RunInfo objects to display
        summary_fields: List of (field_key, display_name) tuples for summary metrics to show
        sparkline_fields: List of (field_key, display_name) tuples for metrics to show as sparklines
        metric_cache: Optional metric history cache for sparkline data
    """

    # Create table with padding between columns
    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Idle", style="dim yellow", width=8)
    table.add_column("Progress", style="yellow", width=25)

    # Add columns for each summary field
    for _, display_name in summary_fields:
        table.add_column(display_name, style="blue", width=15)

    # Add columns for sparklines if requested with extra spacing
    if sparkline_fields:
        for _, display_name in sparkline_fields:
            table.add_column(f"📈 {display_name}", style="cyan", width=25)

    table.add_column("Cost", style="green")

    for run in runs:
        # Format run ID with clickable link to WandB
        wandb_url = f"https://wandb.ai/metta-research/metta/runs/{run.run_id}"
        run_id_text = Text(run.run_id, style="link " + wandb_url)

        # Calculate idle time - use last_updated_at if available, otherwise use created_at
        # Pass created_at to cap idle time to job age
        idle_time = format_idle_time(run.last_updated_at or run.created_at, run.created_at)

        # Format progress in Gsteps with percentage
        if run.total_timesteps and run.current_steps is not None:
            total_gsteps = run.total_timesteps / 1_000_000_000
            current_gsteps = run.current_steps / 1_000_000_000
            percentage = (run.current_steps / run.total_timesteps) * 100
            # Show a space before the unit and format total to 2 decimals
            progress_str = f"{current_gsteps:.2f}/{total_gsteps:.2f} Gsteps ({percentage:.0f}%)"
        else:
            progress_str = "N/A"

        # Format summary fields from run.summary
        field_values = []
        for field_key, _ in summary_fields:
            if run.summary and field_key in run.summary:
                field_value = run.summary[field_key]
                if field_value is not None:
                    if isinstance(field_value, (int, float)):
                        field_str = f"{field_value:.3f}"
                    else:
                        field_str = str(field_value)
                else:
                    field_str = "N/A"
            else:
                field_str = "N/A"
            field_values.append(field_str)

        # Generate sparklines if requested
        sparkline_values = []
        if sparkline_fields and metric_cache:
            # Request metrics to be fetched in background if not cached
            metric_keys = [field_key for field_key, _ in sparkline_fields]
            metric_cache.request_metrics(run.run_id, metric_keys)

            # Get cached metrics (non-blocking)
            history = metric_cache.get_metrics(run.run_id)
            is_fetching = metric_cache.is_fetching(run.run_id)

            for field_key, _ in sparkline_fields:
                # Get current value from summary
                current_value = None
                if run.summary and field_key in run.summary:
                    current_value = run.summary[field_key]

                # Get historical values for sparkline
                values = history.get(field_key, [])
                sparkline = generate_sparkline(values, is_loading=is_fetching, width=15)

                # Format current value with sparkline
                if current_value is not None:
                    if isinstance(current_value, (int, float)):
                        value_str = f"{current_value:.3f}"
                    else:
                        value_str = str(current_value)[:6]
                    combined = f"{value_str:<7} {sparkline}"
                else:
                    combined = f"{'---':<7} {sparkline}"

                sparkline_values.append(combined)
        elif sparkline_fields:
            # No cache provided, show dashed lines with no value
            sparkline_values = [f"{'---':<7} {'─' * 15}"] * len(sparkline_fields)

        # Format cost
        cost_str = f"${run.cost:.2f}" if run.cost else "$0.00"

        # Status with color
        status = run.status
        status_color = _get_status_color(status)
        status_text = Text(status.value, style=status_color)

        table.add_row(
            run_id_text,
            status_text,
            idle_time,
            progress_str,
            *field_values,  # Unpack all summary field values
            *sparkline_values,  # Unpack sparkline values if any
            cost_str,
        )

    return table


def create_run_banner(
    group: Optional[str],
    name_filter: Optional[str],
    runs: List[RunInfo],
    display_limit: int = 10,
    summary_fields: List[tuple[str, str]] | None = None,
    sparkline_fields: List[tuple[str, str]] | None = None,
    api_tracker: Optional[ApiRequestTracker] = None,
):
    """Create a banner with run information."""

    # Calculate runtime from earliest run created_at
    earliest_created = None
    for run in runs:
        if run.created_at:
            # Parse created_at if it's a string from WandB
            created_at = run.created_at
            if isinstance(created_at, str):
                created_at = parser.parse(created_at)

            if earliest_created is None or created_at < earliest_created:
                earliest_created = created_at

    if earliest_created:
        # Use timezone-aware current time to match WandB timestamps
        current_time = datetime.now(timezone.utc) if earliest_created.tzinfo else datetime.now()
        runtime = current_time - earliest_created
        runtime_hours = runtime.total_seconds() / 3600.0
        runtime_str = f"{runtime_hours:.1f} hours"
    else:
        runtime_str = "Unknown"

    # Count runs by status - handle JobStatus enum

    status_counts = {}
    for run in runs:
        # Get the enum value name without the prefix
        status = run.status.name if hasattr(run.status, "name") else str(run.status)
        status_counts[status] = status_counts.get(status, 0) + 1

    total_runs = len(runs)
    completed_runs = status_counts.get("COMPLETED", 0)
    in_training = status_counts.get("IN_TRAINING", 0)
    pending = status_counts.get("PENDING", 0)
    failed = status_counts.get("FAILED", 0)

    # Calculate total cost
    total_cost = sum(run.cost for run in runs)

    # Create filter description
    filter_parts = []
    if group:
        filter_parts.append(f"group={group}")
    if name_filter:
        filter_parts.append(f"name~{name_filter}")

    if filter_parts:
        filter_desc = " | ".join(filter_parts)
    else:
        filter_desc = "all runs"

    # Create inline banner with styled parts

    # Format summary fields display
    field_parts = []
    if summary_fields:
        summary_display = ", ".join([display_name for _, display_name in summary_fields])
        field_parts.append(f"Values: {summary_display}")

    if sparkline_fields:
        sparkline_display = ", ".join([display_name for _, display_name in sparkline_fields])
        field_parts.append(f"Sparklines: {sparkline_display}")

    fields_display = " | ".join(field_parts) if field_parts else "No fields selected"

    # First line with fetch/display info
    line1 = RichText(
        f"🔄 LIVE RUN MONITOR: {filter_desc} | Fetched: {total_runs} runs, "
        f"displaying at most {display_limit} runs | Fields: {fields_display}. "
    )
    line1.append("Use --help to change limits.", style="dim")

    # Cost line with warning
    cost_line = RichText(f"💰 Total Cost: ${total_cost:.2f} ")

    # API stats line
    api_stats_line = ""
    if api_tracker:
        total_requests, requests_per_min = api_tracker.get_stats()
        api_stats_line = f"📡 WandB API: {total_requests} total requests | {requests_per_min:.1f} req/min"

    banner_lines = [
        line1,
        f"⏱️  Runtime: {runtime_str}",
        f"📊 Runs: {total_runs} total | ✅ {completed_runs} completed | 🔄 {in_training} training"
        f" | ⏳ {pending} pending | ❌ {failed} failed",
        cost_line,
    ]

    if api_stats_line:
        banner_lines.append(api_stats_line)

    banner_lines.extend(
        [
            f"🔄 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "─" * 100,
        ]
    )

    return Group(*[Text(line) if isinstance(line, str) else line for line in banner_lines])


def live_monitor_runs(
    group: Optional[str] = None,
    name_filter: Optional[str] = None,
    refresh_interval: int = 30,
    entity: str = "metta-research",
    project: str = "metta",
    clear_screen: bool = True,
    display_limit: int = 10,
    fetch_limit: int = 50,
    summary_fields: List[tuple[str, str]] | None = None,
    sparkline_fields: List[tuple[str, str]] | None = None,
) -> None:
    """Live monitor runs with rich terminal display.

    Args:
        fetch_limit: Maximum number of runs to fetch from WandB (default: 50)
        display_limit: Maximum number of runs to display in table (default: 10)
        summary_fields: List of (field_key, display_name) tuples for summary metrics to show
        sparkline_fields: List of (field_key, display_name) tuples for metrics to show as sparklines
    """

    # Default to showing env_agent/heart.get as Score if no fields specified
    if summary_fields is None:
        summary_fields = [("env_agent/heart.get", "Score")]

    console = Console()

    try:
        from metta.sweep.stores.wandb import WandbStore

        store = WandbStore(entity=entity, project=project)
    except ImportError:
        print("Error: Cannot import WandbStore. Make sure wandb is installed and configured.")
        sys.exit(1)

    # Create API request tracker
    api_tracker = ApiRequestTracker()

    # Create metric cache if sparklines are requested
    metric_cache = None
    if sparkline_fields:
        metric_cache = MetricHistoryCache(entity, project, api_tracker)
        metric_cache.start()

    def generate_display():
        try:
            # Build filters
            filters = {}
            if group:
                filters["group"] = group
            if name_filter:
                filters["name"] = {"regex": name_filter}

            # Fetch runs (already sorted by created_at newest first from WandB)
            # Record API request for fetching runs
            api_tracker.record_request(1)
            all_runs = store.fetch_runs(filters, limit=fetch_limit)
            # Take only the display_limit for the table
            runs = all_runs[:display_limit]

            if not runs:
                # Show warning but keep monitoring
                warning_msg = "⚠️  No runs found matching filters"
                if group or name_filter:
                    filter_parts = []
                    if group:
                        filter_parts.append(f"group={group}")
                    if name_filter:
                        filter_parts.append(f"name~{name_filter}")
                    filter_desc = " | ".join(filter_parts)
                    warning_msg += f": {filter_desc}"

                warning_msg += "\n   Waiting for runs to appear..."

                return Text(warning_msg, style="bright_yellow")

            # Create banner using all fetched runs for accurate statistics
            banner = create_run_banner(
                group, name_filter, all_runs, display_limit, summary_fields, sparkline_fields, api_tracker
            )

            # Create table
            table = make_rich_monitor_table(runs, summary_fields, sparkline_fields, metric_cache)

            # Create a renderable group
            display = Group(
                banner,
                Text(""),  # Empty line
                table,
                Text(""),  # Empty line
                Text("(CMD + Click to see run in WandB)", style="dim"),
            )
            return display

        except Exception as e:
            error_msg = f"Error fetching run data: {str(e)}"
            logger.error(error_msg)
            return Text(error_msg, style="bright_red")

    # Start monitoring
    filter_desc_parts = []
    if group:
        filter_desc_parts.append(f"group={group}")
    if name_filter:
        filter_desc_parts.append(f"name~{name_filter}")

    if filter_desc_parts:
        filter_desc = " | ".join(filter_desc_parts)
        console.print(f"Starting live monitor for runs matching: {filter_desc}")
    else:
        console.print("Starting live monitor for most recent runs")

    console.print(f"Refresh interval: {refresh_interval} seconds")
    console.print(f"Fetch limit: {fetch_limit} runs, Display limit: {display_limit} runs")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(generate_display(), console=console, refresh_per_second=1, screen=clear_screen) as live:
            while True:
                time.sleep(refresh_interval)
                live.update(generate_display())

    except KeyboardInterrupt:
        console.print("\n\nMonitoring stopped by user.")
    except Exception as e:
        console.print(f"\n\nError during monitoring: {e}")
        raise
    finally:
        # Clean up the metric cache
        if metric_cache:
            metric_cache.stop()


def parse_field_spec(field_spec: str) -> tuple[str, str]:
    """Parse a field specification string into field key and display name.

    Args:
        field_spec: Field specification in format 'field_key:display_name' or just 'field_key'

    Returns:
        Tuple of (field_key, display_name)
    """
    field_spec = field_spec.strip()
    if ":" in field_spec:
        field_key, display_name = field_spec.split(":", 1)
    else:
        # If no display name specified, create one from the field key
        field_key = field_spec
        # Try to create a reasonable display name from the field key
        # e.g., "env_agent/heart.get" -> "Heart"
        parts = field_key.replace("/", ".").split(".")
        display_name = parts[-2].title() if len(parts) > 1 else field_key
    return (field_key.strip(), display_name.strip())


def main():
    """CLI entry point for live run monitoring."""
    parser = argparse.ArgumentParser(
        description="Live monitor runs with rich terminal display",
        epilog="""
Examples:
  %(prog)s --group my_group_name
  %(prog)s --name-filter "axel.*"
  %(prog)s --group my_group --name-filter "experiment.*"
  %(prog)s --refresh 15 --entity myteam --project myproject
  %(prog)s --fetch-limit 100 --display-limit 20  # Fetch more, display more
  %(prog)s  # Monitor last 10 runs (fetch 50, display 10)

The monitor will display a live table with color-coded statuses:
  - Completed runs (blue)
  - In training runs (green)
  - Pending runs (gray)
  - Training done, no eval (orange)
  - Failed runs (red)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", "-g", help="WandB group to monitor (optional)")
    parser.add_argument("--name-filter", help="Regex filter for run names (e.g., 'axel.*')")
    parser.add_argument("--refresh", "-r", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument(
        "--entity", "-e", type=str, default="metta-research", help="WandB entity (default: metta-research)"
    )
    parser.add_argument("--project", "-p", type=str, default="metta", help="WandB project (default: metta)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen, append output instead")
    parser.add_argument(
        "--fetch-limit", type=int, default=50, help="Maximum number of runs to fetch from WandB (default: 50)"
    )
    parser.add_argument(
        "--display-limit", type=int, default=10, help="Maximum number of runs to display in table (default: 10)"
    )
    parser.add_argument(
        "--summary-fields",
        type=str,
        default="env_agent/heart.get:Score",
        help="Comma-separated list of summary fields to display. Format: 'field_key:display_name' or just 'field_key'. "
        "Example: 'env_agent/heart.get:Score,env_agent/food.get:Food,train/loss:Loss' "
        "(default: env_agent/heart.get:Score)",
    )
    parser.add_argument(
        "--sparklines",
        type=str,
        help="Comma-separated list of metrics to display as sparklines. "
        "Format: 'field_key:display_name' or just 'field_key'. "
        "Example: 'train/loss:Loss,eval/accuracy:Acc' "
        "These will show mini time-series graphs in the table.",
    )

    args = parser.parse_args()

    # Parse summary fields
    summary_fields = []
    if args.summary_fields:
        for field_spec in args.summary_fields.split(","):
            summary_fields.append(parse_field_spec(field_spec))

    # Parse sparkline fields
    sparkline_fields = []
    if args.sparklines:
        for field_spec in args.sparklines.split(","):
            sparkline_fields.append(parse_field_spec(field_spec))

    # Validate refresh interval
    if args.refresh < 1:
        print("Error: Refresh interval must be at least 1 second")
        sys.exit(1)

    # Validate limits
    if args.fetch_limit < 1:
        print("Error: Fetch limit must be at least 1")
        sys.exit(1)
    if args.display_limit < 1:
        print("Error: Display limit must be at least 1")
        sys.exit(1)
    if args.display_limit > args.fetch_limit:
        print("Warning: Display limit is greater than fetch limit, some runs may not be shown")
        print(f"  Fetch limit: {args.fetch_limit}, Display limit: {args.display_limit}")

    # Validate WandB access
    try:
        import wandb

        if not wandb.api.api_key:
            print("Error: WandB API key not found. Please run 'wandb login' first.")
            sys.exit(1)
    except ImportError:
        print("Error: WandB not installed. Please install with 'pip install wandb'.")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not validate WandB credentials: {e}")

    try:
        live_monitor_runs(
            group=args.group,
            name_filter=args.name_filter,
            refresh_interval=args.refresh,
            entity=args.entity,
            project=args.project,
            clear_screen=not args.no_clear,
            display_limit=args.display_limit,
            fetch_limit=args.fetch_limit,
            summary_fields=summary_fields,
            sparkline_fields=sparkline_fields,
        )
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
