#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///
"""
Expand test matrix configuration for GitHub Actions.

Takes a simple configuration of packages with shard counts and expands it
into a full matrix configuration for parallel test execution.
"""

import json
import sys
from typing import Any


def expand_test_matrix(config: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Expand a test configuration into a full matrix.

    Args:
        config: List of package configurations with 'name', 'package', and 'shards'

    Returns:
        A dictionary with 'include' key containing the expanded matrix entries
    """
    matrix_entries = []

    for item in config:
        name = item["name"]
        package = item.get("package", name)  # Default package to name if not specified
        shards = item.get("shards", 1)

        if shards == 1:
            # No sharding needed
            entry = {"name": name, "package": package}
            # Include any additional fields from the original config
            for key, value in item.items():
                if key not in ["name", "package", "shards"]:
                    entry[key] = value
            matrix_entries.append(entry)
        else:
            # Create an entry for each shard
            for shard_id in range(shards):
                entry = {
                    "name": f"{name}-shard-{shard_id}",
                    "package": package,
                    "shard_id": shard_id,
                    "num_shards": shards,
                }
                # Include any additional fields from the original config
                for key, value in item.items():
                    if key not in ["name", "package", "shards"]:
                        entry[key] = value
                matrix_entries.append(entry)

    return {"include": matrix_entries}


def main():
    """Main entry point for the script."""
    # The configuration is passed as a JSON string via environment variable
    #
    # example:
    #
    # config = [
    #     {"name": "agent", "package": "agent", "shards": 1},
    #     {"name": "common", "package": "common", "shards": 1},
    #     {"name": "app_backend", "package": "app_backend", "shards": 3},
    #     {"name": "mettagrid", "package": "mettagrid", "shards": 2},
    #     {"name": "core", "package": "core", "shards": 7},
    # ]

    # Allow configuration to be passed via stdin
    if not sys.stdin.isatty():
        try:
            config = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON input: {e}", file=sys.stderr)
            sys.exit(1)

    # Expand the matrix
    matrix = expand_test_matrix(config)

    # Output the matrix in the format GitHub Actions expects
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
