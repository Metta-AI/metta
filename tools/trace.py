# Generate a graphical trace of multiple runs.

import os
import subprocess
import platform
import hydra
from omegaconf import OmegaConf
from agent.policy_store import PolicyStore
from rl.wandb.wandb_context import WandbContext
from rl.pufferlib.trace import save_trace_image, save_replay
from util.runtime_configuration import setup_metta_environment
from util.config import config_from_path


@hydra.main(version_base=None, config_path="../configs", config_name="simulator")
def main(cfg):

    setup_metta_environment(cfg)

    env_cfg = config_from_path(cfg.env, cfg.env_overrides)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_record = policy_store.policy(cfg.policy_uri)

        image_path = f"{cfg.run_dir}/traces/trace.png"
        save_trace_image(cfg, env_cfg, policy_record, image_path)
        if platform.system() == "Darwin":
            # Open image in Preview.
            subprocess.run(["open", image_path])

        replay_path = f"{cfg.run_dir}/replays/replay.json"
        save_replay(cfg, env_cfg, policy_record, replay_path)

if __name__ == "__main__":
    main()
