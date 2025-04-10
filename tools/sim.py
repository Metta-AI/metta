import logging

import hydra
from omegaconf import DictConfig
from util.runtime_configuration import setup_metta_environment
from rl.wandb.wandb_context import WandbContext
from eval import simulate_policy

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)
    with WandbContext(cfg) as wandb_run:
        simulate_policy(cfg, wandb_run)

if __name__ == "__main__":
    main()
