import json
import logging
import os
import tempfile
import zlib
from typing import Optional

import wandb
from omegaconf import DictConfig
from wandb.sdk import wandb_run

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.file import write_file

logger = logging.getLogger(__name__)


def upload_map_preview(
    env_config: DictConfig,
    s3_path: Optional[str] = None,
    wandb_run: Optional[wandb_run.Run] = None,
):
    """
    Builds a map preview of the simulation environment and uploads it to S3.

    Args:
        cfg: Configuration for the simulation
        s3_path: Path to upload the map preview to
        wandb_run: Weights & Biases run object for logging
    """
    logger.info("Building map preview...")

    env = MettaGridEnv(env_config, render_mode=None)

    preview = {
        "version": 1,
        "action_names": env.action_names(),
        "object_types": env.object_type_names(),
        "inventory_items": env.inventory_item_names(),
        "map_size": [env.map_width, env.map_height],
        "num_agents": env.num_agents,
        "max_steps": 1,
        "grid_objects": list(env.grid_objects.values()),
    }

    # Compress data with deflate
    preview_data = json.dumps(preview)  # Convert to JSON string
    preview_bytes = preview_data.encode("utf-8")  # Encode to bytes
    compressed_data = zlib.compress(preview_bytes)  # Compress the bytes

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Create directory and save compressed file
        preview_path = temp_file.name
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        with open(preview_path, "wb") as f:
            f.write(compressed_data)

    if s3_path is None:
        logger.info("No S3 path provided, skipping upload")
        return

    # Upload to S3 using our new utility function
    try:
        write_file(path=s3_path, local_file=preview_path,
                   content_type="application/x-compress")
    except Exception as e:
        logger.error(f"Failed to upload preview map to S3: {str(e)}")

    try:
        # If upload was successful, log the link to WandB
        if wandb_run:
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={s3_path}"
            link_summary = {
                "replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Map Preview</a>')}
            wandb_run.log(link_summary)
            logger.info(f"Preview map available at: {player_url}")
    except Exception as e:
        logger.error(f"Failed to log preview map to WandB: {str(e)}")
