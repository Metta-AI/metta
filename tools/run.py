import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c
from rl.wandb.wandb import init_wandb
from rl.carbs.carb_sweep import run_sweep
from util.stats import print_policy_stats

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    framework = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)
    if cfg.wandb.track:
        init_wandb(cfg)

    try:
        if cfg.cmd == "train":
            framework.train()

        if cfg.cmd == "evaluate":
            policy_stats = framework.evaluate()
            print_policy_stats(policy_stats)

        if cfg.cmd == "play":
            framework.evaluate()

        if cfg.cmd == "sweep":
            run_sweep(cfg)

    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
