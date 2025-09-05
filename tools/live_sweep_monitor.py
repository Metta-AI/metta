#!/usr/bin/env uv run
"""Live sweep monitor with rich terminal display.

This utility provides a real-time, color-coded view of sweep runs with automatic refresh.

Features:
- Color-coded status: Completed (blue), In Training (green), Pending (gray),
  Training Done No Eval (orange), Failed (red)
- Live progress updates in Gsteps (billions of steps)
- Cost tracking in USD format
- Auto-refresh every 30 seconds (configurable)
- In-place updates (no scrolling output)
- Runs sorted with completed runs at bottom

Usage:
    ./tools/live_sweep_monitor.py my_sweep_name
    ./tools/live_sweep_monitor.py my_sweep_name --refresh 15
    ./tools/live_sweep_monitor.py my_sweep_name --entity myteam --project myproject
"""

import argparse
import sys
from pathlib import Path

# Add the metta-repo root to the path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from metta.sweep.utils import live_monitor_sweep


def main():
    """CLI entry point for live sweep monitoring."""
    parser = argparse.ArgumentParser(
        description="Live monitor a sweep with rich terminal display",
        epilog="""
Examples:
  %(prog)s my_sweep_name
  %(prog)s my_sweep_name --refresh 15
  %(prog)s my_sweep_name --entity myteam --project myproject

The monitor will display a live table with color-coded statuses:
  - Completed runs (blue) at bottom
  - In training runs (green)
  - Pending runs (gray)
  - Training done, no eval (orange)
  - Failed runs (red)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sweep_id", help="Sweep ID to monitor")
    parser.add_argument(
        "--refresh", "-r",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--entity", "-e",
        type=str,
        default="metta-research",
        help="WandB entity (default: metta-research)"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default="metta",
        help="WandB project (default: metta)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock data (no WandB connection required)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear screen, append output instead"
    )

    args = parser.parse_args()

    # Validate refresh interval
    if args.refresh < 1:
        print("Error: Refresh interval must be at least 1 second")
        sys.exit(1)

    # Test mode with mock data
    if args.test:
        from metta.sweep.utils import live_monitor_sweep_test
        try:
            live_monitor_sweep_test(
                sweep_id=args.sweep_id,
                refresh_interval=args.refresh,
                clear_screen=not args.no_clear
            )
        except KeyboardInterrupt:
            print("\nTest monitoring stopped by user.")
            sys.exit(0)
        return

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
        live_monitor_sweep(
            sweep_id=args.sweep_id,
            refresh_interval=args.refresh,
            entity=args.entity,
            project=args.project,
            clear_screen=not args.no_clear
        )
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
