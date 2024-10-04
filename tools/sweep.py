import os
import signal  # Aggressively exit on ctrl+c

import hydra
import wandb
from omegaconf import OmegaConf
from rich import traceback
from rl.carbs.rollout import CarbsSweepRollout
from rl.carbs.sweep import CarbsSweep, create_sweep_state_if_needed
from util.seeding import seed_everything

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

global _cfg
global _consecutive_failures
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    global _cfg
    _cfg = cfg

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    create_sweep_state_if_needed(cfg)

    with CarbsSweep(cfg.run_dir) as sweep_state:
        wandb_sweep_id = sweep_state.wandb_sweep_id

    global _consecutive_failures
    _consecutive_failures = 0

    wandb.agent(wandb_sweep_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                function=run_carb_sweep_rollout,
                count=999999)

def run_carb_sweep_rollout():
    global _consecutive_failures
    global _cfg

    if _consecutive_failures > 10:
        print("Too many consecutive failures, exiting")
        os._exit(0)

    try:
        rollout = CarbsSweepRollout(_cfg)
        if rollout.run():
            _consecutive_failures = 0
        else:
            _consecutive_failures += 1
    except Exception as e:
        _consecutive_failures += 1
        raise e


if __name__ == "__main__":
    main()
