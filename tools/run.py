import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c
from rl.wandb.wandb import init_wandb
from rl.carbs.carb_sweep import run_sweep
from rl.pufferlib.train import train
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.play import play

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))

    if cfg.wandb.track:
        init_wandb(cfg)

    try:
        if cfg.cmd == "train":
            train(cfg)

        if cfg.cmd == "evaluate":
            evaluate(cfg)

        if cfg.cmd == "play":
            play(cfg)

        if cfg.cmd == "sweep":
            run_sweep(cfg)

    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
