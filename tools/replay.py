# Generate a graphical trace of multiple runs.

import os
import subprocess
import platform
import hydra
from omegaconf import OmegaConf
from agent.policy_store import PolicyStore
from rl.wandb.wandb_context import WandbContext
from rl.pufferlib.replay_helper import ReplayHelper
from util.runtime_configuration import setup_mettagrid_environment
from util.config import config_from_path


@hydra.main(version_base=None, config_path="../configs", config_name="simulator")
def main(cfg):

    setup_mettagrid_environment(cfg)

    env_cfg = config_from_path(cfg.env, cfg.env_overrides)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_record = policy_store.policy(cfg.policy_uri)
        replay_helper = ReplayHelper(cfg, env_cfg, policy_record, wandb_run)
        replay_path = f"{cfg.run_dir}/replays/replay.json.z"
        replay_helper.generate_replay(replay_path)
        print(f"Replay saved to {replay_path}")

if __name__ == "__main__":
    main()
