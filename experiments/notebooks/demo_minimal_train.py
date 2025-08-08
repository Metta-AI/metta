#!/usr/bin/env -S uv run
"""Minimal script to test training cell logic from Hello-World notebook."""

import subprocess
from datetime import datetime

import yaml
from metta.common.util.fs import get_repo_root
from metta.interface.environment import _get_default_env_config
from omegaconf import OmegaConf

# 1. Build hallway environment config
hallway_map = """###########
#@.......m#
###########"""

env_dict = _get_default_env_config(num_agents=1, width=11, height=3)
env_dict["game"]["map_builder"] = {
    "_target_": "metta.map.mapgen.MapGen",
    "border_width": 0,
    "root": {
        "type": "metta.map.scenes.inline_ascii.InlineAscii",
        "params": {"data": hallway_map},
    },
}
mine = env_dict["game"]["objects"]["mine_red"]
mine["initial_resource_count"] = 1
mine["conversion_ticks"] = 4
mine["cooldown"] = 0
mine["max_output"] = 2
env_dict["game"]["agent"]["rewards"]["inventory"]["ore_red"] = 1.0

cfg = OmegaConf.create(
    {
        "env": env_dict,
        "renderer_job": {
            "policy_type": "opportunistic",
            "num_steps": 100,
            "num_agents": 1,
            "sleep_time": 0.04,
        },
    }
)

# 2. Write minimal curriculum YAML
repo_root = get_repo_root()
cfg_tmp_dir = repo_root / "configs" / "tmp"
cfg_tmp_dir.mkdir(parents=True, exist_ok=True)
cur_name = f"hello_world_curriculum_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml"
cur_path = cfg_tmp_dir / cur_name
with open(cur_path, "w") as f:
    yaml.dump(
        {
            "_pre_built_env_config": env_dict,
            "game": env_dict["game"],
            "name": "hallway_curriculum",
        },
        f,
        default_flow_style=False,
        indent=2,
    )

# 3. Launch training
run_name = f"hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
train_py = str(repo_root / "tools" / "train.py")
train_cmd = [
    train_py,
    f"run={run_name}",
    f"trainer.curriculum=tmp/{cur_name}",
    "wandb=off",
    "device=cpu",
    "trainer.total_timesteps=10000",
    "trainer.batch_size=256",
    "trainer.minibatch_size=256",
    "trainer.num_workers=2",
    "sim=sim",
    "+train_job.evals.name=hallway",
    "+train_job.evals.num_episodes=1",
    "+train_job.evals.simulations={}",
    "trainer.simulation.evaluate_interval=0",
]
print("Starting training (∼30s)…\n")
proc = subprocess.Popen(
    train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
for line in proc.stdout or []:
    print(line, end="")
proc.wait()

# 4. Cleanup and report
tmp = cur_path
cur_path.unlink(missing_ok=True)
print(f"\nTraining complete – checkpoints saved under train_dir/{run_name}")
