# Generate a graphical trace of multiple runs.

import os
import subprocess
import hydra
from omegaconf import OmegaConf
from rl.wandb.wandb_context import WandbContext
from mettagrid.config.config import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.pufferlib.trace import save_trace_image


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    setup_metta_environment(cfg)

    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_record = policy_store.policy(cfg.policy_uri)
        image_path = f"{cfg.run_dir}/traces/trace.png"
        save_trace_image(cfg, policy_record, image_path)
        # Open image in Finder
        subprocess.run(["open", image_path])

if __name__ == "__main__":
    main()
