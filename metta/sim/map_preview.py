import json
import os
import zlib
from logging import Logger

import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb.sdk import wandb_run

from metta.util.config import config_from_path
from metta.util.s3 import upload_file
from mettagrid.mettagrid_env import MettaGridEnv


def upload_map_preview(cfg: DictConfig | ListConfig, wandb_run: wandb_run.Run, logger: Logger):
    """
    Builds a map preview of the simulation environment and uploads it to S3.
    Skips upload when running in CI environments.

    Args:
        cfg: Configuration for the simulation
        wandb_run: Weights & Biases run object for logging
        logger: Logger object for logging messages
    """
    logger.info("Building map preview...")

    logger.info(f"cfg: {cfg}")

    env_path = cfg.trainer.env
    env_cfg = config_from_path(env_path)

    # MettaGridEnv requires a DictConfig
    env_dict = OmegaConf.to_container(env_cfg)
    env = MettaGridEnv(DictConfig(env_dict), render_mode=None)

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

    # Create directory and save compressed file
    preview_path = f"{cfg.run_dir}/replays/replay.0.json.z"
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    with open(preview_path, "wb") as f:
        f.write(compressed_data)

    # Upload to S3 using our new utility function
    try:
        preview_url = f"replays/{cfg.run}/replay.0.json.z"
        s3_url = upload_file(
            file_path=preview_path,
            s3_key=preview_url,
            content_type="application/x-compress",
            logger=logger,
            skip_if_ci=True,
        )

        # If upload was successful, log the link to WandB
        if s3_url:
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={s3_url}"
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Map Preview</a>')}
            wandb_run.log(link_summary)
            logger.info(f"Preview map available at: {player_url}")

    except Exception as e:
        logger.error(f"Failed to upload preview map to S3: {str(e)}")
