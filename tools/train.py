import json
import logging
import os
import sys
import zlib

import boto3
import hydra
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from rich.logging import RichHandler
from torch.distributed.elastic.multiprocessing.errors import record

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import Config, config_from_path, setup_metta_environment
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

# Configure rich colored logging
logging.basicConfig(
    level="INFO", format="%(processName)s %(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("train")


# TODO: populate this more
class TrainJob(Config):
    evals: SimulationSuiteConfig


def train(cfg, wandb_run):
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        logger.info(f"Loading train config overrides from {overrides_path}")
        override_cfg = OmegaConf.load(overrides_path)

        # Set struct flag to False to allow accessing undefined fields
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, override_cfg)
        # Optionally, restore struct behavior after merge
        OmegaConf.set_struct(cfg, True)

    if os.environ.get("RANK", "0") == "0":
        with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_job = TrainJob(cfg.train_job)

    policy_store = PolicyStore(cfg, wandb_run)

    if "map_preview" in cfg and cfg.map_preview:
        # the user wants to preview the map (python -m tools.train run=... trainer.env=... +map_preview=true )
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

    if "dry_run" in cfg and cfg.dry_run:
        return

    trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store, train_job.evals)
    trainer.train()
    trainer.close()


@record
@hydra.main(config_path="../configs", config_name="train_job", version_base=None)
def main(cfg: OmegaConf) -> int:
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)
    logger.info(f"Train job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    logger.info(
        f"Training {cfg.run} on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    if "LOCAL_RANK" in os.environ and cfg.device.startswith("cuda"):
        logger.info(f"Initializing distributed training with {os.environ['LOCAL_RANK']} {cfg.device}")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.device = f"{cfg.device}:{local_rank}"
        dist.init_process_group(backend="nccl")

    logger.info(f"Training {cfg.run} on {cfg.device}")
    if os.environ.get("RANK", "0") == "0":
        with WandbContext(cfg, job_type="train") as wandb_run:
            train(cfg, wandb_run)
    else:
        train(cfg, None)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
