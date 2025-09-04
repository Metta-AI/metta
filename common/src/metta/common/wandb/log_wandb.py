#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
# ]
# ///
"""
Command-line interface for logging values to wandb.
"""

import argparse
import json
import logging
import sys

from metta.common.wandb.utils import log_debug_info, log_single_value

logger = logging.getLogger(__name__)


def auto_type(value_str: str):
    """
    Parse value from command line.
    Tries JSON for complex types, otherwise returns the string.
    """
    try:
        # This handles dicts, lists, bools, null, numbers
        return json.loads(value_str)
    except json.JSONDecodeError:
        # Just return as string
        return value_str


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Log values to wandb",
        epilog="Examples:\n"
        "  %(prog)s my/metric 42\n"
        "  %(prog)s accuracy 0.95 --step 1000\n"
        "  echo 3.14 | %(prog)s accuracy/train\n"
        "  %(prog)s --debug\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("key", nargs="?", help="Metric key to log")
    parser.add_argument("value", nargs="?", help="Value to log (reads from stdin if not provided)")
    parser.add_argument("--step", type=int, default=0, help="Step to log at (default: 0)")
    parser.add_argument("--no-summary", action="store_true", help="Don't add to wandb summary")
    parser.add_argument("--debug", action="store_true", help="Log debug environment info")

    args = parser.parse_args()

    try:
        if args.debug:
            log_debug_info()
        else:
            if not args.key:
                parser.error("Key is required unless using --debug")

            # Get value from args or stdin
            if args.value is not None:
                value_str = args.value
            else:
                # Read from stdin
                value_str = sys.stdin.read().strip()
                if not value_str:
                    raise RuntimeError("No value provided (empty stdin)")

            # Parse value
            value = auto_type(value_str)

            # Log to wandb
            log_single_value(args.key, value, step=args.step, also_summary=not args.no_summary)

            # Echo value to stdout for chaining
            print(value)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
