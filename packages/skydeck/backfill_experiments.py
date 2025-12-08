#!/usr/bin/env python3
"""Backfill experiments table from currently running jobs."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from skydeck.database import Database
from skydeck.models import DesiredState


async def backfill_experiments():
    """Create experiment records from running jobs."""
    db_path = str(Path.home() / ".skydeck" / "skydeck.db")
    db = Database(db_path)

    try:
        await db.connect()

        # Get all running jobs
        jobs = await db.get_running_jobs()
        print(f"Found {len(jobs)} running jobs")

        for job in jobs:
            # Check if experiment already exists
            existing = await db.get_experiment(job.experiment_id)
            if existing:
                print(f"Experiment {job.experiment_id} already exists, skipping")
                continue

            # Parse command to extract base_command and flags
            command = job.command
            base_command = "lt"  # default
            flags = {}

            if command:
                # Extract flags from command
                # Format: ./devops/skypilot/launch.py --nodes=4 --gpus=4 ...
                # recipes.experiment.cog_arena.train key=value key=value ...
                parts = command.split()

                # Find module path (e.g., recipes.experiment.cog_arena.train)
                module_idx = None
                for i, part in enumerate(parts):
                    if part.startswith("recipes.") or part.startswith("common."):
                        module_idx = i
                        break

                if module_idx:
                    # Everything after the module path are flags
                    for flag_part in parts[module_idx + 1 :]:
                        if "=" in flag_part:
                            key, value = flag_part.split("=", 1)
                            # Skip special flags
                            if key not in ["run", "--skip-git-check"]:
                                # Parse value type
                                if value.lower() == "true":
                                    flags[key] = True
                                elif value.lower() == "false":
                                    flags[key] = False
                                elif value.isdigit():
                                    flags[key] = int(value)
                                else:
                                    flags[key] = value

            # Create experiment
            from datetime import UTC, datetime

            now = datetime.now(UTC).isoformat()
            await db._conn.execute(
                """
                INSERT INTO experiments (
                    id, name, base_command, run_name, nodes, gpus,
                    instance_type, cloud, region, spot, flags,
                    desired_state, current_state, current_job_id,
                    cluster_name, tags, description,
                    exp_group, exp_order, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.experiment_id,  # id
                    job.experiment_id,  # name
                    base_command,
                    job.experiment_id,  # run_name
                    job.nodes,
                    job.gpus,
                    job.instance_type,
                    job.cloud,
                    job.region,
                    False,  # spot
                    json.dumps(flags),  # flags as JSON string
                    DesiredState.RUNNING.value,
                    job.status.value,
                    job.id,
                    job.cluster_name,  # cluster_name
                    json.dumps([]),  # tags as empty array
                    "",  # description
                    None,  # exp_group
                    0,  # exp_order
                    now,  # created_at
                    now,  # updated_at
                ),
            )
            await db._conn.commit()

            print(f"Created experiment: {job.experiment_id}")
            print(f"  Status: {job.status.value}")
            print(f"  Flags: {flags}")

        print("\nBackfill complete!")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(backfill_experiments())
