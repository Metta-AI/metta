import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c
from rl.wandb.wandb import init_wandb
from rl.carbs.carb_sweep import run_sweep
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.play import play
from rich.console import Console
from rl.pufferlib.train import PufferTrainer
import random
import numpy as np
import torch

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)
    init_wandb(cfg)

    try:
        if cfg.cmd == "train":
            trainer = PufferTrainer(cfg)
            trainer.train()

        if cfg.cmd == "evaluate":
            evaluate(cfg)

        if cfg.cmd == "play":
            play(cfg)

        if cfg.cmd == "sweep":
            run_sweep(cfg)

    except KeyboardInterrupt:
        os._exit(0)
    except Exception:
        Console().print_exception()
        os._exit(0)


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

if __name__ == "__main__":
    main()
