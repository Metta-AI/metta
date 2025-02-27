import logging
import os
import hydra
from agent.policy_store import PolicyStore
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rl.wandb.wandb_context import WandbContext
import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(processName)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True
)

logger = logging.getLogger("train")

@record
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    local_rank = int(os.environ["LOCAL_RANK"])
    cfg.device = f'{cfg.device}:{local_rank}'
    dist.init_process_group(backend="nccl")

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        trainer = hydra.utils.instantiate(cfg.trainer, cfg, wandb_run, policy_store)
        trainer.train()
        trainer.close()

if __name__ == "__main__":
    main()
