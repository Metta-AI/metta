import os
import sys
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c
from rl.carbs.carb_sweep import run_sweep
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.play import play
from rl.pufferlib.dashboard.dashboard import Dashboard
from rich.console import Console
from rl.pufferlib.train import PufferTrainer
import random
import numpy as np
import torch
from util.stats import print_policy_stats
from io import StringIO
from rl.pufferlib.dashboard.utilization import Utilization
from rl.pufferlib.dashboard.logs import Logs
from rl.pufferlib.dashboard.wandb import WanDb
from rl.pufferlib.dashboard.trainer import Trainer
import rich

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.data_dir, cfg.experiment), exist_ok=True)
    stdout_path = os.path.join(cfg.data_dir, cfg.experiment, 'out.log')
    stderr_path = os.path.join(cfg.data_dir, cfg.experiment, 'error.log')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout = open(stdout_path, 'w')
    stderr = open(stderr_path, 'w')
    sys.stderr = stderr
    sys.stdout = stdout

    traceback.install(show_locals=False)

    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    dashboard = None
    error = False
    try:
        if cfg.cmd == "train":
            trainer = PufferTrainer(cfg)
            dashboard = Dashboard(cfg, components=[
                Utilization(),
                WanDb(cfg.wandb),
                Trainer(trainer),
                Logs(stdout_path, stderr_path),
            ])
            trainer.train()
            trainer.close()

        if cfg.cmd == "evaluate":
            dashboard = Dashboard(cfg, components=[
                Utilization(),
                WanDb(cfg.wandb),
                Logs(stdout_path, stderr_path),
            ])
            stats = evaluate(cfg)
            print_policy_stats(stats)

        if cfg.cmd == "play":
            play(cfg)

        if cfg.cmd == "sweep":
            run_sweep(cfg)

    except KeyboardInterrupt:
        print("Ctrl+C detected, exiting...")
        os._exit(0)
    except Exception:
        error = sys.exc_info()
        Console(file=stderr).print_exception(
            show_locals=True,
            extra_lines=3,
            suppress=[]
        )
    finally:
        if dashboard is not None:
            dashboard.stop()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout.close()
        stderr.close()

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
