"""Shared display utilities for job and run monitoring.

This module provides common formatting and display functions used across
different monitoring tools (experiment monitor, run monitor, job monitor).
"""

from datetime import datetime, timedelta

# Status symbols used across monitoring tools
STATUS_SYMBOLS = {
    "completed": "✓",
    "succeeded": "✓",
    "failed": "✗",
    "running": "⋯",
    "pending": "○",
    "cancelled": "✗",
}


def get_status_symbol(status: str) -> str:
    """Get symbol for job/run status.

    Args:
        status: Status string (completed, failed, running, pending, etc.)

    Returns:
        Unicode symbol representing the status
    """
    return STATUS_SYMBOLS.get(status.lower(), "○")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Examples:
        42.5 -> "42s"
        125.3 -> "2m 5s"
        3725.8 -> "1h 2m 5s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"

    delta = timedelta(seconds=int(seconds))
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if delta.days > 0:
        parts.append(f"{delta.days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display.

    Example: "2024-01-15 14:30:22"
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_gsteps(steps: int | float) -> str:
    """Format steps in billions (Gsteps).

    Args:
        steps: Number of timesteps

    Returns:
        Formatted string like "1.23 Gsteps"
    """
    gsteps = steps / 1_000_000_000
    return f"{gsteps:.2f} Gsteps"


def format_cost(cost: float) -> str:
    """Format cost in USD.

    Args:
        cost: Cost in dollars

    Returns:
        Formatted string like "$12.34"
    """
    return f"${cost:.2f}"
