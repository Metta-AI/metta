import json
import os
import zlib
from logging import Logger

import boto3
import wandb
from omegaconf import DictConfig, ListConfig
from wandb.sdk import wandb_run

from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path


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

    sim = cfg.sim
    env_cfg = config_from_path(sim.env)

    try:
        vecenv = make_vecenv(env_cfg, sim.vectorization, num_envs=1, render_mode="human")

        # Check if vecenv was created successfully
        if vecenv is None or len(vecenv.envs) == 0:
            logger.error("Failed to create vector environment")
            return

        env = vecenv.envs[0]

        preview = {
            "version": 1,
            "action_names": env.action_names(),
            "object_types": env.object_type_names(),
            "inventory_items": env.inventory_item_names(),
            "map_size": [env.map_width, env.map_height],
            "num_agents": vecenv.num_agents,
            "max_steps": 1,
            "grid_objects": list(env.grid_objects.values()),
        }

        # Clear any actions
        for obj in preview["grid_objects"]:
            if "action" in obj:
                del obj["action"]

        # Compress data with deflate
        preview_data = json.dumps(preview)  # Convert to JSON string
        preview_bytes = preview_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(preview_bytes)  # Compress the bytes

        # Create directory and save compressed data
        preview_path = f"{cfg.run_dir}/replays/replay.0.json.z"
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        with open(preview_path, "wb") as f:
            f.write(compressed_data)

        # Upload to S3
        try:
            # Check if running in CI environment - abort upload if in CI
            is_ci = os.environ.get("CI", "").lower() in ("1", "true", "yes")
            if is_ci:
                logger.info("Running in CI environment, skipping S3 upload but keeping local file")
                return

            preview_url = f"replays/{cfg.run}/replay.0.json.z"
            s3_client = boto3.client("s3")
            s3_bucket = "softmax-public"

            s3_client.upload_file(
                Filename=preview_path,
                Bucket=s3_bucket,
                Key=preview_url,
                ExtraArgs={"ContentType": "application/x-compress"},
            )

            # Log the link to WandB
            player_url = f"https://metta-ai.github.io/metta/?replayUrl=https://{s3_bucket}.s3.us-east-1.amazonaws.com/{preview_url}"
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Map Preview</a>')}
            wandb_run.log(link_summary)
            logger.info(f"Preview map available at: {player_url}")

        except Exception as e:
            logger.error(f"Failed to upload preview map to S3: {str(e)}")

    except Exception as e:
        logger.error(f"Error building map preview: {str(e)}")

    finally:
        # Clean up environment resources if needed
        if "vecenv" in locals() and vecenv is not None:
            try:
                vecenv.close()
            except Exception as e:
                logger.warning(f"Error closing vector environment: {str(e)}")
