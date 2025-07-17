#!/usr/bin/env -S uv run
"""
A script to download a stats file from wandb or S3 and launch duckdb against it.

Usage:
    ./tools/stats_duckdb_cli.py ++eval_db_uri=wandb://stats/my_stats_db
"""

import logging
import subprocess

import hydra
from omegaconf import DictConfig

from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid.util.file import local_copy
from metta.util.metta_script import setup_mettagrid_environment

logger = logging.getLogger("stats_duckdb_cli")


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


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_job")
def main(cfg: DictConfig) -> int:
    """
    Main function to download a stats file and launch duckdb against it.
    """
    setup_mettagrid_environment(cfg)
    setup_mettagrid_logger()
    # Check if eval_db_uri is configured
    if not hasattr(cfg, "eval_db_uri") or not cfg.eval_db_uri:
        logger.error("Error: eval_db_uri is not configured")
        print("Please set eval_db_uri in your configuration or use the command line override.")
        print("Example: ./tools/stats_duckdb_cli.py +eval_db_uri=wandb://stats/navigation_db")
        return 1

    uri = cfg.eval_db_uri

    # Validate URI format
    if not (uri.startswith("wandb://") or uri.startswith("s3://")):
        logger.error(f"Error: URI must start with wandb:// or s3://, got {uri}")
        return 1

    try:
        # Use the local_copy context manager to get a local path
        with local_copy(uri) as local_path:
            stats_db = EvalStatsDB(local_path)
            stats_db.con.commit()
            stats_db.close()
            logger.info(f"Downloaded to temporary location: {local_path}")
            launch_duckdb_cli(local_path)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
