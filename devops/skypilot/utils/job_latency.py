#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""
Calculate SkyPilot queue latency from SKYPILOT_TASK_ID env var.

Expected format: sky-YYYY-MM-DD-HH-MM-SS-ffffff_<cluster>_<n>
"""

import datetime
import logging
import os
import re
import sys
from typing import Final

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.util.log_config import init_logging
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.common.wandb.utils import log_to_wandb_summary
from mettagrid.base_config import Config

_EPOCH: Final = datetime.timezone.utc
_FMT: Final = "%Y-%m-%d-%H-%M-%S-%f"

# Regex for SkyPilot task ID format
_TS_RE: Final = re.compile(r"^sky(?:-managed)?-(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6,9})_")


def parse_submission_timestamp(task_id: str) -> datetime.datetime:
    m = _TS_RE.match(task_id)
    if not m:
        raise ValueError(f"Invalid task ID format: {task_id}")

    # Truncate to 26 chars to ensure exactly 6 digits for microseconds
    ts_part = m.group("ts")[:26]  # YYYY-MM-DD-HH-MM-SS-ffffff (26 chars)
    try:
        return datetime.datetime.strptime(ts_part, _FMT).replace(tzinfo=_EPOCH)
    except ValueError as e:
        raise ValueError(f"Failed to parse timestamp from task ID: {task_id}") from e


def calculate_queue_latency() -> float:
    """Calculate SkyPilot queue latency in seconds."""
    task_id = os.environ.get("SKYPILOT_TASK_ID")
    if not task_id:
        raise RuntimeError("SKYPILOT_TASK_ID environment variable not set")

    submitted = parse_submission_timestamp(task_id)
    return (datetime.datetime.now(_EPOCH) - submitted).total_seconds()


if __name__ == "__main__":
    init_logging()
    logger = logging.getLogger("metta_agent")

    script_start_time = datetime.datetime.now(_EPOCH).isoformat()
    task_id = os.environ.get("SKYPILOT_TASK_ID", "unknown")

    metrics = {
        "skypilot/latency_script_ran": True,
        "skypilot/latency_script_time": script_start_time,
        "skypilot/task_id": task_id,
    }

    exit_code = 0

    try:
        latency_sec = calculate_queue_latency()
        logger.info(f"SkyPilot queue latency: {latency_sec:.1f} s (task: {task_id})")

        metrics.update(
            {
                "skypilot/queue_latency_s": latency_sec,
                "skypilot/latency_calculated": True,
            }
        )

    except Exception as e:
        logger.error(f"SkyPilot queue latency: N/A (task_id: {task_id}, error: {e})")
        metrics.update(
            {
                "skypilot/latency_calculated": False,
                "skypilot/latency_error": str(e),
            }
        )
        exit_code = 1

    finally:
        wandb_config = WandbConfig(
            enabled=True,
            project=os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY", METTA_WANDB_ENTITY),
            run_id=os.environ.get("METTA_RUN_ID"),
            job_type="skypilot_latency",
            tags=["skypilot", "latency"],
        )

        try:
            with WandbContext(wandb_config, Config(), timeout=15) as run:
                if run:
                    log_to_wandb_summary(metrics)
                    logger.info(f"Logged metrics to W&B run: {run.id}")
                else:
                    logger.warning("W&B run not initialized (offline or no connection)")
        except Exception as wandb_err:
            logger.warning(f"W&B logging failed: {wandb_err}")

        sys.exit(exit_code)
