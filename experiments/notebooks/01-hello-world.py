# ruff: noqa
# fmt: off
# %% [markdown]
# # Hello World: your first metta-ai experiment
#
# Welcome to your first reinforcement learning experiment in the metta-ai project. This notebook will guide you through creating, observing, evaluating, and training AI agents in a simple gridworld environment.
#
# ## What You'll Learn
#
# By the end of this notebook, you'll be able to:
# - Create and understand ASCII maps
# - Choose and observe different agent policies
# - Evaluate agent performance quantitatively
# - Train a new agent from scratch
# - Compare the performance of two agents
#
# ## 1. Setup
#
# Let's load dependencies and set up some scaffolding. Don't worry about the details here.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Setup imports for core notebook workflow
# %load_ext autoreload
# %autoreload 2

import time
import warnings
import io, contextlib
import os, json, subprocess, tempfile, yaml
from pathlib import Path
from datetime import datetime

import numpy as np  # used later
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Any, Dict  # type: ignore
from metta.common.util.fs import get_repo_root
from tools.renderer import setup_environment, get_policy
import ipywidgets as widgets
from IPython.display import display

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


# %% [markdown]
# ## 2. Defining an Environment
#
# In Metta AI, an **environment** is the virtual world where our agents act and learn. It has 2 main elements:
#
#   1. A Map -- the physical layout of the environment where agents can move and what objects they encounter. One simple way to define a map is to use a simple ASCII string. This much ASCII will get us started:
#       - `#` = walls that block movement
#       - `@` = where the agent starts
#       - `.` = empty spaces where agents can walk
#       - `m` = a mine that generates collectible ore
#   2. Game Rules -- what actions are available and how rewards are calculated. We express these rules through environment configuration.
#
# For now, we'll mostly rely on the default set of game rules, as follows:
#
# **Agents Can Observe:**
# - **Vision**: Agents see an 11x11 grid around themselves
# - **Awareness**: Agents know what resources they're carrying
# - **Feedback**: Agents receive information about their last action's success

# **Agents Can Act:**
# - Navigate in 8 directions (cardinal + diagonal) and rotate
# - Pick up & carry resources from objects like mines, or drop items
# - Interact with other agents -- we'll ignore this for now and just start with a single agent
#
# **Agents Encounter Objects:**
# - **Walls**: Block movement and create boundaries
# - **Mines**: Automatically generate ore over time
# - **Ore**: Collectible resources that agents can carry and trade for rewards
# - **Rewards**: Collecting ore gives small positive rewards that can be used to reinforce behavior
#
# In our environment, we'll also need an agent (sometimes referred to as a "policy") who can take action.
# We'll try out a very simple "opportunistic" agent that moves randomly around the environment. If it encounters
# a resource, it will usually (but not always) pick it up.
#
# In the following cell we'll lay out the map in ASCII, configure the environment to use it, and choose the opportunistic agent. We'll also set:
# - How many steps to run the simulation for
# - How long to sleep between steps
# - Some other basic parameters
# %%
# Define ASCII map and environment configuration
hallway_map = """###########
#@.......m#
###########"""

env_cfg = get_cfg("benchmark")  # type: ignore
# Convert to plain dict so we can edit
env_dict: Dict[str, Any] = OmegaConf.to_container(env_cfg, resolve=True)  # type: ignore
# Override for a single 11x3 hallway map
env_dict["game"]["num_agents"] = 1  # type: ignore
env_dict["game"]["obs_width"] = 11  # type: ignore
env_dict["game"]["obs_height"] = 11  # type: ignore
env_dict["game"]["map_builder"] = {
    "_target_": "mettagrid.mapgen.mapgen.MapGen",
    "border_width": 0,
    "root": {
        "type": "mettagrid.mapgen.scenes.inline_ascii.InlineAscii",
        "params": {"data": hallway_map},
    },
}
env_dict["game"]["objects"]["mine_red"]["initial_resource_count"] = 1
env_dict["game"]["objects"]["mine_red"]["conversion_ticks"] = 4
env_dict["game"]["objects"]["mine_red"]["cooldown"] = 0
env_dict["game"]["objects"]["mine_red"]["max_output"] = 2  # type: ignore
env_dict["game"]["objects"]["mine_red"]["max_conversions"] = -1  # type: ignore
env_dict["game"]["objects"]["generator_red"]["max_conversions"] = -1  # type: ignore
env_dict["game"]["agent"]["rewards"]["inventory"]["ore_red"] = 1.0

cfg = OmegaConf.create({
    "env": env_dict,
    "renderer_job": {
        "policy_type": "opportunistic",
        "num_steps": 100,
        "num_agents": 1,
        "sleep_time": 0.04,
    },
})

# %% [markdown]
# ## 3. Observing a Simulation
#
#  Now we'll actually run the simulation, using a "game loop" approach, where we:
# - Find out what action the agent wants to take
# - Step the environment forward one tick, taking the action into account
# - Render the environment to the screen (as an ASCII string)
# - Sleep for a bit
#
# We'll also track the agent's inventory and display the score.
#
# %%
with contextlib.redirect_stdout(io.StringIO()):
    env, render_mode = setup_environment(cfg)
    policy = get_policy(cfg.renderer_job.policy_type, env, cfg)

header = widgets.HTML()
map_box = widgets.HTML()
display(header, map_box)

# Run simulation loop
obs, info = env.reset()
for step in range(cfg.renderer_job.num_steps):
    actions = policy.predict(obs)
    obs, rewards, terminals, truncations, info = env.step(actions)
    # Track ore in inventory for agent 0
    agent_obj = next(o for o in env.grid_objects.values() if o.get("agent_id") == 0)
    inv = {env.resource_names[idx]: count for idx, count in agent_obj.get("inventory", {}).items()}
    header.value = f"<b>Step:</b> {step+1}/{cfg.renderer_job.num_steps} <br/> <b>Inventory:</b> {inv}"
    with contextlib.redirect_stdout(io.StringIO()):
        buffer_str = env.render()
    map_box.value = f"<pre>{buffer_str}</pre>"
    if cfg.renderer_job.sleep_time:
        time.sleep(cfg.renderer_job.sleep_time)

env.close()
# %% [markdown]
# ### What You Should See:
# - The agent (`0`) moving back and forth randomly in the hallway
# - The mine ('m') is continually generating ore (not shown)
# - When the agent reaches the mine, it should sometimes pick up ore
# - This will increase the agent's "score"
#
#
# ## 4. Evaluation – defining “success” for our hallway task
#
# So far we've just watched the agent wander. Now we need a **quantitative** way to decide whether any
# policy is "good".
#
# ### 4.1 Desired behavior
# • Reach the red mine and harvest as much red ore as possible.
# • Do it quickly – fewer steps means more ore before the episode ends.
#
# ### 4.2 Choosing a metric
# The simplest measurable signal that captures that behavior is **how much `ore_red` the agent is carrying when the episode ends**.
#
# We therefore define:
#
#     score = total amount of `ore_red` in the agent's inventory
#
# Why this is a good choice:
# 1. **Direct** – it counts exactly the thing we care about.
# 2. **Monotonic** – more ore ⇒ higher score.
# 3. **Reward-friendly** – the environment can hand out a small reward each time inventory grows, which is useful later when we train.
#
# ### 4.3 Hooking the metric into the config
# Metta-ai's env config already supports inventory-based rewards. We enable it with:
#
# ```yaml
# game:
#   agent:
#     rewards:
#       inventory:
#         ore_red: 1.0        # +1 for every unit of red ore held
# ```
#
# During an episode the environment sums that reward, so the **episode return** equals the final ore count. That value is what we'll call *score*.
#
# ### 4.4 Evaluation procedure
# 1. Run *N* episodes (default 100) with the current policy.
# 2. Record the episode return (our *score*) after each run.
# 3. Report the mean and standard deviation.
#
# The same procedure works for any future policy, giving a fair apples-to-apples comparison.
#
# When you run the next code cell you'll see a table with:
# - episode index
# - score for that episode
# - running average
#
# This establishes a numeric baseline for the opportunistic agent. Later we'll train a policy and expect this number to rise significantly.

# %%
EVAL_EPISODES = 30
scores: list[int] = []

# Re-use the same cfg (contains ore_red reward = 1.0)
with contextlib.redirect_stdout(io.StringIO()):
    eval_env, _ = setup_environment(cfg)
    eval_policy = get_policy(cfg.renderer_job.policy_type, eval_env, cfg)

for ep in range(1, EVAL_EPISODES + 1):
    obs, _ = eval_env.reset()
    # Run fixed number of steps to make scores comparable to the observation cell
    inv_count = 0
    for step in range(cfg.renderer_job.num_steps):
        actions = eval_policy.predict(obs)
        obs, _, _, _, _ = eval_env.step(actions)
    # After the episode, check inventory
    agent_obj = next(o for o in eval_env.grid_objects.values() if o.get("agent_id") == 0)
    inv = {eval_env.resource_names[idx]: cnt for idx, cnt in agent_obj.get("inventory", {}).items()}
    inv_count = int(inv.get("ore_red", 0))
    scores.append(inv_count)
    print(f"Episode {ep:3d}/{EVAL_EPISODES}: ore_red = {inv_count}")

mean_score = np.mean(scores)
std_score = np.std(scores)
print("\n=== Summary ===")
print(f"Mean ore_red: {mean_score:.2f} ± {std_score:.2f} (n={EVAL_EPISODES})")

# Display table inline
running_avg = pd.Series(scores).expanding().mean()
display(
    pd.DataFrame({"episode": list(range(1, EVAL_EPISODES + 1)), "ore_red": scores, "running_avg": running_avg})
)

eval_env.close()

# %% [markdown]
# ## 5. Training a New Agent
#
# We've measured how well the *hand-coded* opportunistic policy performs. Now we'll teach an agent **from scratch** using
# reinforcement learning (RL) and see if it can beat that baseline.
#
# ### 5.1  What does "training" mean?
# In RL the agent initially acts at random. After each step the environment returns a *reward*. Over many episodes the
# learning algorithm (we'll use PPO – *Proximal Policy Optimization*) updates the policy so that actions leading to higher
# cumulative reward become more likely.
#
# In our hallway task the reward signal is already in place: every unit of `ore_red` in inventory is worth **+1**.
# Maximizing reward therefore means collecting as much ore as possible.
#
# ### 5.2  Minimal training configuration
# A full-scale run might take millions of timesteps; for demonstration we'll run a *tiny* job just to prove the pipeline:
# - same hallway environment (so results stay comparable)
# - 10 000 environment steps on CPU (≈30 s)
# - checkpoints & logs saved under `train_dir/`
#
# ### 5.3  Launching training
# The repo provides `tools/train.py` – a thin CLI around the trainer. We pass it:
# 1. a unique run name (`run=`)
# 2. an inline curriculum file that simply references our hallway config
# 3. overrides (`trainer.total_timesteps`, etc.) to keep it small.
#
# Feel free to increase `trainer.total_timesteps` later for a stronger agent.

# %%
cfg_tmp_dir = get_repo_root() / "configs" / "tmp"
cfg_tmp_dir.mkdir(parents=True, exist_ok=True)

curriculum_name = f"hello_world_curriculum_{datetime.now():%Y%m%d_%H%M%S}.yaml"
temp_curriculum_path = cfg_tmp_dir / curriculum_name

with temp_curriculum_path.open("w") as f:
    yaml.dump(
        {
            "_pre_built_mg_config": env_dict,
            "game": env_dict["game"],
            "name": "hallway_curriculum",
        },
        f,
        default_flow_style=False,
        indent=2,
    )

# Unique run name (so multiple notebook runs don't collide)
run_name = f"hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Build command
repo_root = get_repo_root()
train_cmd = [
    str(repo_root / "tools" / "train.py"),
    f"run={run_name}",
    f"training_env.curriculum=tmp/{curriculum_name}",
    "wandb=off",
    "device=cpu",
    "trainer.total_timesteps=10000",  # tiny demo run
    "trainer.batch_size=256",
    "trainer.minibatch_size=256",
    "trainer.num_workers=2",
    "sim=sim",
    "+train_job.evals.name=hallway","+train_job.evals.num_episodes=1","+train_job.evals.simulations={}",
]

process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout or []:
    print(line, end="")
process.wait()

temp_curriculum_path.unlink(missing_ok=True)

# %% [markdown]
# ## 6. Understanding Training Results
#
# - **Logs** live in `train_dir/{run_name}/*.log`
# - **Checkpoints** (PyTorch `.pt` files) are in `train_dir/{run_name}/checkpoints/`
#   the latest one is the policy we’ll load next.
# - **Replays** (optional) would be in `train_dir/{run_name}/replays/`
#
# You can inspect the logs or open a checkpoint later to see the learned network weights.

# %% [markdown]
# ## 7. Observing the Trained Agent
#
# Let’s load the newest checkpoint and watch the trained policy in the same hallway environment.

# %%
# Locate latest checkpoint
ckpt_dir = Path("train_dir") / run_name / "checkpoints"
latest_ckpt = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
print("Loading", latest_ckpt.name)

# Build cfg for trained policy
auto_cfg = OmegaConf.create({
    "env": env_dict,
    "policy_uri": f"file://{latest_ckpt.absolute()}",
    "renderer_job": {
        "policy_type": "trained",
        "num_steps": 100,
        "num_agents": 1,
        "sleep_time": 0.04,
    },
})

with contextlib.redirect_stdout(io.StringIO()):
    trained_env, _ = setup_environment(auto_cfg)
    trained_policy = get_policy("trained", trained_env, auto_cfg)

header2 = widgets.HTML()
map_box2 = widgets.HTML()
display(header2, map_box2)

obs, _ = trained_env.reset()
for step in range(auto_cfg.renderer_job.num_steps):
    actions = trained_policy.predict(obs)
    obs, _, _, _, _ = trained_env.step(actions)
    agent_obj = next(o for o in trained_env.grid_objects.values() if o.get("agent_id") == 0)
    inv = {trained_env.resource_names[i]: c for i, c in agent_obj.get("inventory", {}).items()}
    header2.value = f"<b>Step:</b> {step+1}/{auto_cfg.renderer_job.num_steps} <br/> <b>Inventory:</b> {inv}"
    with contextlib.redirect_stdout(io.StringIO()):
        buf = trained_env.render()
    map_box2.value = f"<pre>{buf}</pre>"
    if auto_cfg.renderer_job.sleep_time:
        time.sleep(auto_cfg.renderer_job.sleep_time)

trained_env.close()
