import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))

    try:
        from rl.carbs.carb_sweep import run_sweep
        run_sweep(cfg)

    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
