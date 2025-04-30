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
    logger.info("Building map preview...")

    sim = cfg.sim

    """Generate a minimal replay for mettascope."""
    env_cfg = config_from_path(sim.env)
    vecenv = make_vecenv(env_cfg, sim.vectorization, num_envs=1, render_mode="human")
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

    # Compress it with deflate.
    preview_data = json.dumps(preview)  # Convert to JSON string
    preview_bytes = preview_data.encode("utf-8")  # Encode to bytes
    compressed_data = zlib.compress(preview_bytes)  # Compress the bytes

    preview_path = f"{cfg.run_dir}/replays/replay.0.json.z"
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)

    with open(preview_path, "wb") as f:
        f.write(compressed_data)

    preview_url = f"replays/{cfg.run}/replay.0.json.z"

    s3_client = boto3.client("s3")
    s3_bucket = "softmax-public"
    s3_client.upload_file(
        Filename=preview_path,
        Bucket=s3_bucket,
        Key=preview_url,
        ExtraArgs={"ContentType": "application/x-compress"},
    )
    link = f"https://{s3_bucket}.s3.us-east-1.amazonaws.com/{preview_url}"
    # Log the link to WandB
    player_url = "https://metta-ai.github.io/metta/?replayUrl=" + link
    link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Map Preview</a>')}
    wandb_run.log(link_summary)

    logger.info(f"preview map @ {player_url}")
