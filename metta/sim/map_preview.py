import json
import logging
import os
import tempfile
import zlib
from typing import Optional

import wandb
from wandb.sdk import wandb_run

from metta.common.util.constants import METTASCOPE_REPLAY_URL_PREFIX
from metta.common.util.fs import get_repo_root
from metta.utils.file import write_file
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator import Simulation, Simulator

logger = logging.getLogger(__name__)


def write_map_preview_file(preview_path: str, sim: Simulation, gzipped: bool):
    logger.info("Building map preview...")

    preview = {
        "version": 1,
        "action_names": sim.action_names,
        "object_types": sim.object_type_names,
        "inventory_items": sim.resource_names,
        "map_size": [sim.map_width, sim.map_height],
        "num_agents": sim.num_agents,
        "max_steps": 1,
        "grid_objects": list(sim.grid_objects().values()),
    }

    preview_data = json.dumps(preview).encode("utf-8")  # Convert to JSON string
    if gzipped:
        # Compress data with deflate
        preview_data = zlib.compress(preview_data)

    with open(preview_path, "wb") as f:
        f.write(preview_data)


def write_local_map_preview(sim: Simulation):
    repo_root = get_repo_root()
    maps_dir = repo_root / "outputs" / "maps"
    os.makedirs(maps_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=maps_dir, suffix=".json") as temp_file:
        preview_path = os.path.relpath(temp_file.name)

        # no gzip locally - fastapi doesn't recognize .json.z files
        write_map_preview_file(preview_path, sim, gzipped=False)

    return preview_path


def upload_map_preview(
    s3_path: str,
    env_cfg: MettaGridConfig,
    wandb_run: Optional[wandb_run.Run] = None,
):
    """
    Builds a map preview of the simulation environment and uploads it to S3.

    Args:
        cfg: Configuration for the simulation
        s3_path: Path to upload the map preview to
        wandb_run: Weights & Biases run object for logging
    """

    simulator = Simulator()
    sim = simulator.new_simulation(env_cfg)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Create directory and save compressed file
        preview_path = temp_file.name
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        write_map_preview_file(preview_path, sim, gzipped=True)

    # Upload to S3 using our new utility function
    try:
        write_file(path=s3_path, local_file=preview_path, content_type="application/x-compress")
    except Exception as e:
        logger.error(f"Failed to upload preview map to S3: {str(e)}", exc_info=True)

    try:
        # If upload was successful, log the link to WandB
        if wandb_run:
            player_url = f"{METTASCOPE_REPLAY_URL_PREFIX}{s3_path}"
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Map Preview</a>')}
            wandb_run.log(link_summary)
            logger.info(f"Preview map available at: {player_url}")
    except Exception as e:
        logger.error(f"Failed to log preview map to WandB: {str(e)}", exc_info=True)
