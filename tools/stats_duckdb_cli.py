#!/usr/bin/env python3
"""
A script to download a stats file from wandb or S3 and launch duckdb against it.

Usage:
    python stats_duckdb.py wandb://stats/evals_jack_testing
    python stats_duckdb.py s3://my-bucket/path/to/stats.db
"""

import argparse
import logging
import subprocess

from mettagrid.util.file import local_copy


def launch_duckdb_cli(file_path):
    """
    Launch duckdb CLI against the specified file.

    Args:
        file_path: Path to the file to open in duckdb
    """
    # Check if duckdb is installed
    try:
        subprocess.run(["duckdb", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: duckdb CLI is not installed or not in PATH.")
        print("Please install duckdb: https://duckdb.org/docs/installation/")
        return False

    # Launch duckdb with the file
    print(f"\nLaunching duckdb with file: {file_path}\n")
    print("=" * 60)
    print("Type .exit to quit duckdb")
    print("=" * 60)

    return subprocess.call(["duckdb", str(file_path)], shell=False)


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Download a stats file and launch duckdb against it")
    parser.add_argument("uri", help="URI in format wandb://project/artifact_name[:version] or s3://bucket/path/to/file")
    args = parser.parse_args()

    # Validate URI format
    if not (args.uri.startswith("wandb://") or args.uri.startswith("s3://")):
        print("Error: URI must start with wandb:// or s3://")
        return 1

    try:
        # Use the local_copy context manager to get a local path
        with local_copy(args.uri) as local_path:
            print(f"Downloaded to temporary location: {local_path}")
            launch_duckdb_cli(local_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
