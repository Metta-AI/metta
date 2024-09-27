import os
import random
import signal  # Aggressively exit on ctrl+c
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from rich import traceback
from rich.console import Console

from rl.carbs.carbs_controller import CarbsController
from rl.pufferlib.dashboard.dashboard import Dashboard
from rl.pufferlib.dashboard.logs import Logs
from rl.pufferlib.dashboard.policy import Policy
from rl.pufferlib.dashboard.training import Training
from rl.pufferlib.dashboard.utilization import Utilization
from rl.pufferlib.dashboard.wandb import WanDb
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.dashboard.carbs import Carbs
from rl.pufferlib.play import play
from rl.pufferlib.train import PufferTrainer
from util.stats import print_policy_stats
from util.logging import remap_io, restore_io
from rl.wandb.wandb import init_wandb
import wandb
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

os.environ["WANDB_SILENT"] = "true"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    logs_path = os.path.join(cfg.data_dir, cfg.experiment)
    remap_io(logs_path)
    traceback.install(show_locals=False)

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    dashboard = None
    error = False
    try:
        if cfg.cmd == "train":
            init_wandb(cfg)
            trainer = PufferTrainer(cfg)
            dashboard = Dashboard(cfg, components=[
                Utilization(),
                WanDb(cfg.wandb),
                Training(trainer),
                Policy(trainer.policy_checkpoint),
                Logs(logs_path),
            ])
            trainer.train()
            trainer.close()
            wandb.finish(quiet=True)
        if cfg.cmd == "evaluate":
            dashboard = Dashboard(cfg, components=[
                Utilization(),
                WanDb(cfg.wandb),
                Logs(logs_path),
            ])
            stats = evaluate(cfg)
            print_policy_stats(stats)

        if cfg.cmd == "play":
            play(cfg)

        if cfg.cmd == "sweep":
            carbs_controller = CarbsController(cfg)
            dashboard = Dashboard(cfg, components=[
                Utilization(),
                WanDb(cfg.wandb),
                Carbs(carbs_controller),
                Logs(logs_path),
            ])
            carbs_controller.run_sweep()

    except KeyboardInterrupt:
        print("Ctrl+C detected, exiting...")
        restore_io()
        os._exit(0)
    except Exception:
        error = sys.exc_info()
        Console().print_exception(
            show_locals=True,
            extra_lines=3,
            suppress=[]
        )
    finally:
        if dashboard is not None:
            dashboard.stop()

    restore_io()
    if error:
        tb = traceback.Traceback.from_exception(
                    *error,
                    # show_locals=True,
                    extra_lines=3,
                    suppress=[]
        )
        Console().print(tb)
        os._exit(1)
        # with open(stderr_path, 'a') as f:
        #     Console(file=f).print(tb)

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

if __name__ == "__main__":
    main()
