import os
import signal  # Aggressively exit on ctrl+c

import hydra
from agent.policy_store import PolicyStore
from util.runtime_configuration import setup_metta_environment
from omegaconf import OmegaConf
from rl.pufferlib.play import play
from rl.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)

    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        play(cfg, policy_store)


if __name__ == "__main__":
    main()
