import os
import signal  # Aggressively exit on ctrl+c

import hydra
from omegaconf import OmegaConf
from rich import traceback
from rl.pufferlib.play import play
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)

    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
   
    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy = policy_store.policy(cfg.evaluator.policy)

        play(cfg, policy)


if __name__ == "__main__":
    main()
