import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", app_title="Hello metta-ai")


@app.cell
def _():
    # ruff: noqa
    # fmt: off
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Hello World: your first metta-ai experiment

    Welcome to your first reinforcement learning experiment in the metta-ai project. This notebook will guide you through creating, observing, evaluating, and training AI agents in a simple gridworld environment.

    ## What You'll Learn

    By the end of this notebook, you'll be able to:

    - Make a simple game map and rules -- an "environment" for running RL experiments
    - Watch agents explore your game map
    - Evaluate agent performance quantitatively
    - Train a new agent from scratch
    - Compare the performance of two agents

    ## 1. Setup

    Let's load dependencies and set up some scaffolding. Don't worry about the details here.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Setup imports for core notebook workflow
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    import time
    import warnings
    import io, contextlib
    import os, json, subprocess, tempfile, yaml
    from datetime import datetime

    import numpy as np  # used later
    import pandas as pd
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from typing import Any, Dict  # type: ignore
    from metta.mettagrid.util.hydra import get_cfg  # type: ignore
    from metta.common.util.fs import get_repo_root
    from tools.renderer import setup_environment, get_policy
    import anywidget
    import traitlets
    from IPython.display import display

    # Define a minimal HTML widget using anywidget so we can drop ipywidgets
    class HTMLWidget(anywidget.AnyWidget):
        """A simple widget that renders arbitrary HTML content.

        The widget keeps a single 'value' trait (a string) that is
        mirrored between Python and the front-end. Any updates to the
        value are immediately reflected in the browser.
        """

        _esm = """
        function render({ model, el }) {
          function update() {
            el.innerHTML = model.get("value") ?? "";
          }
          model.on("change:value", update);
          update();
        }
        export default { render };
        """
        value = traitlets.Unicode("").tag(sync=True)

    class _WidgetsNamespace:
        """Drop-in replacement for the subset of ipywidgets API used here."""

        def HTML(self, value=""):
            return HTMLWidget(value=value)

    widgets = _WidgetsNamespace()

    # Suppress Pydantic deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
    print("Setup done")
    return (
        Any,
        Dict,
        OmegaConf,
        Path,
        contextlib,
        datetime,
        display,
        get_cfg,
        get_policy,
        get_repo_root,
        io,
        mo,
        np,
        pd,
        setup_environment,
        subprocess,
        time,
        widgets,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Defining an Environment

    In Metta AI, an **environment** is the virtual world where our agents act and learn. It has 2 main elements:

      1. **A map** -- the physical layout of the environment where agents can move and what objects they encounter. One simple way to define a map is to use a simple ASCII string. This much ASCII will get us started:
          - `#` = walls that block movement
          - `@` = where the agent starts
          - `.` = empty spaces where agents can walk
          - `m` = a mine that generates collectible ore
      2. **Game rules** -- what actions are available and how rewards are calculated.

    For now, we'll mostly rely on the default set of game rules. We'll also populate our environment with one agent (also known as a "policy"). The basic rules of agents are:

    **Agents Can Observe:**

    - **Vision**: Agents can see around themselves
    - **Awareness**: Agents know what resources they're carrying
    - **Feedback**: Agents receive information about their last action's success

    **Agents Can Act:**

    - Navigate in 8 directions (cardinal + diagonal) and rotate.  (They do face in a specific direction)
    - Pick up & carry (or drop) resources like ore and hearts
    - Interact with other agents -- but for now we'll stick to one agent

    **Agents Encounter Objects:**

    - **Walls**: Block movement and create boundaries
    - **Mines**: Automatically generate ore over time
    - **Ore**: Collectible resources that agents can carry and trade for rewards
    - **Rewards**: Collecting ore gives small positive rewards that can be used to reinforce behavior

    We'll start with a simple "opportunistic" agent that is hard-coded to make random moves around the map. If it encounters a resource, it will usually (but not always) pick it up. That's it.

    In the following cell we'll lay out the map in ASCII, configure the environment to use it, and select the opportunistic agent. We'll also set:

    - How many steps to run the simulation for
    - How long to sleep between steps
    - Some other basic parameters

    Feel free to adjust parameters and see what happens.
    """
    )
    return


@app.cell
def _(Any, Dict, OmegaConf, get_cfg):
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
        "_target_": "metta.map.mapgen.MapGen",
        "border_width": 0,
        "root": {
            "type": "metta.map.scenes.inline_ascii.InlineAscii",
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
    env_dict["game"]["num_observation_tokens"] = (
        200  # Default value expected by MettaAgent
    )

    # Disable+enable to make the environment we need
    env_dict["game"]["actions"]["attack"]["enabled"] = 0
    env_dict["game"]["actions"]["noop"]["enabled"] = 0
    env_dict["game"]["actions"]["move"]["enabled"] = 0
    env_dict["game"]["actions"]["rotate"]["enabled"] = 0
    env_dict["game"]["actions"]["move_cardinal"] = {"enabled": 1}
    env_dict["game"]["actions"]["move_8way"] = {"enabled": 0}
    env_dict["game"]["actions"]["change_color"]["enabled"] = 0
    env_dict["game"]["actions"]["change_glyph"]["enabled"] = 0
    env_dict["game"]["actions"]["swap"]["enabled"] = 0
    env_dict["game"]["actions"]["put_items"]["enabled"] = 0
    env_dict["game"]["actions"]["get_items"]["enabled"] = 1

    cfg = OmegaConf.create(
        {
            "env": env_dict,
            "renderer_job": {
                "policy_type": "opportunistic",
                "num_steps": 200,
                "num_agents": 1,
                "sleep_time": 0.04,
            },
        }
    )
    print("made env")
    return cfg, env_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Observing a Simulation

     Now we'll actually run the simulation, using a "game loop" approach, where we:

    - Find out what action the agent wants to take
    - Step the environment forward one tick, taking the action into account
    - Render the environment to the screen (as an ASCII string)
    - Sleep for a bit

    We'll also track the agent's inventory and display the score.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    observe_button = mo.ui.run_button(label="Click to run observation below")
    observe_button
    return (observe_button,)


@app.cell
def _(
    cfg,
    contextlib,
    display,
    get_policy,
    io,
    mo,
    observe_button,
    setup_environment,
    time,
    widgets,
):
    mo.stop(not observe_button.value)

    with contextlib.redirect_stdout(io.StringIO()):
        env, render_mode = setup_environment(cfg)
        policy = get_policy(cfg.renderer_job.policy_type, env, cfg)
    header = widgets.HTML()
    map_box = widgets.HTML()
    display(header, map_box)
    _obs, info = env.reset()
    for _step in range(cfg.renderer_job.num_steps):
        _actions = policy.predict(_obs)
        _obs, rewards, terminals, truncations, info = env.step(_actions)
        _agent_obj = next(
            (o for o in env.grid_objects.values() if o.get("agent_id") == 0)
        )
        _inv = {
            env.inventory_item_names[idx]: count
            for idx, count in _agent_obj.get("inventory", {}).items()
        }
        header.value = f"<b>Step:</b> {_step + 1}/{cfg.renderer_job.num_steps} <br/> <b>Inventory:</b> {_inv}"
        with contextlib.redirect_stdout(io.StringIO()):
            buffer_str = env.render()
        map_box.value = f"<pre>{buffer_str}</pre>"
        if cfg.renderer_job.sleep_time:
            time.sleep(cfg.renderer_job.sleep_time)
    env.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### What You Should See:

    - The agent (`0`) moving back and forth randomly in the hallway
    - The mine ('m') is continually generating ore (not shown)
    - When the agent reaches the mine, it should sometimes pick up ore
    - This will increase the agent's "score"


    ## 4. Evaluation – defining “success” for our hallway task

    So far we've just watched the agent wander. Now we need a **quantitative** way to decide whether any
    policy is "good".

    ### 4.1 Desired behavior

    - Reach the red mine and harvest as much red ore as possible.
    - Do it quickly – fewer steps means more ore before the episode ends.

    ### 4.2 Choosing a metric

    The simplest measurable signal that captures that behavior is **how much `ore_red` the agent is carrying when the episode ends**.

    We therefore define:

        score = total amount of `ore_red` in the agent's inventory

    Why this is a good choice:

    1. **Direct** – it counts exactly the thing we care about.
    2. **Monotonic** – more ore ⇒ higher score.
    3. **Reward-friendly** – the environment can hand out a small reward each time inventory grows, which is useful later when we train.

    ### 4.3 Hooking the metric into the config
    Metta-ai's env config already supports inventory-based rewards. We enable it with:

    ```yaml
    game:
      agent:
        rewards:
          inventory:
            ore_red: 1.0        # +1 for every unit of red ore held
    ```

    During an episode the environment sums that reward, so the **episode return** equals the final ore count. That value is what we'll call *score*.

    ### 4.4 Evaluation procedure
    1. Run *N* episodes (default 100) with the current policy.
    2. Record the episode return (our *score*) after each run.
    3. Report the mean and standard deviation.

    The same procedure works for any future policy, giving a fair apples-to-apples comparison.

    When you run the next code cell you'll see a table with:
    - episode index
    - score for that episode
    - running average

    This establishes a numeric baseline for the opportunistic agent. Later we'll train a policy and expect this number to rise significantly.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    eval_button = mo.ui.run_button(label="Click to run evaluation below")
    eval_button
    return (eval_button,)


@app.cell
def _(
    cfg,
    contextlib,
    display,
    eval_button,
    get_policy,
    io,
    mo,
    np,
    pd,
    setup_environment,
):
    mo.stop(not eval_button.value)
    EVAL_EPISODES = 10
    scores: list[int] = []
    with contextlib.redirect_stdout(io.StringIO()):
        eval_env, _ = setup_environment(cfg)
        eval_policy = get_policy(cfg.renderer_job.policy_type, eval_env, cfg)
    for ep in range(1, EVAL_EPISODES + 1):
        _obs, _ = eval_env.reset()
        inv_count = 0
        for _step in range(cfg.renderer_job.num_steps):
            _actions = eval_policy.predict(_obs)
            _obs, _, _, _, _ = eval_env.step(_actions)
        _agent_obj = next(
            (o for o in eval_env.grid_objects.values() if o.get("agent_id") == 0)
        )
        _inv = {
            eval_env.inventory_item_names[idx]: cnt
            for idx, cnt in _agent_obj.get("inventory", {}).items()
        }
        inv_count = int(_inv.get("ore_red", 0))
        scores.append(inv_count)
        # print(f"Episode {ep:3d}/{EVAL_EPISODES}: ore_red = {inv_count}")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    # print("\n=== Summary ===")
    # print(f"Mean ore_red: {mean_score:.2f} ± {std_score:.2f} (n={EVAL_EPISODES})")
    running_avg = pd.Series(scores).expanding().mean()
    display(
        pd.DataFrame(
            {
                "episode": list(range(1, EVAL_EPISODES + 1)),
                "ore_red": scores,
                "running_avg": running_avg,
            }
        )
    )
    eval_env.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Training a New Agent

    We've measured how well the *hand-coded* opportunistic policy performs. Now we'll teach an agent **from scratch** using
    reinforcement learning (RL) and see if it can beat that baseline.

    ### 5.1  What does "training" mean?
    In RL the agent initially acts at random. After each step the environment returns a *reward*. Over many episodes the
    learning algorithm (we'll use PPO – *Proximal Policy Optimization*) updates the policy so that actions leading to higher
    cumulative reward become more likely.

    In our hallway task the reward signal is already in place: every unit of `ore_red` in inventory is worth **+1**.
    Maximizing reward therefore means collecting as much ore as possible.

    ### 5.2  Minimal training configuration
    A full-scale run might take millions of timesteps; for demonstration we'll run a *tiny* job just to prove the pipeline:

    - same hallway environment (so results stay comparable)
    - 10 000 environment steps on CPU (≈30 s)
    - checkpoints & logs saved under `train_dir/`

    ### 5.3  Launching training
    The repo provides `tools/train.py` – a thin CLI around the trainer. We pass it:

    1. a unique run name (`run=`)
    2. an inline curriculum file that simply references our hallway config
    3. overrides (`trainer.total_timesteps`, etc.) to keep it small.

    Feel free to increase `trainer.total_timesteps` later for a stronger agent.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Click to run training below")
    train_button
    return (train_button,)


@app.cell
def _(
    datetime,
    env_dict: "Dict[str, Any]",
    get_repo_root,
    mo,
    subprocess,
    train_button,
    yaml,
):
    mo.stop(not train_button.value)
    cfg_tmp_dir = get_repo_root() / "configs" / "tmp"
    cfg_tmp_dir.mkdir(parents=True, exist_ok=True)

    curriculum_name = f"hello_world_curriculum_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    temp_curriculum_path = cfg_tmp_dir / curriculum_name

    with temp_curriculum_path.open("w") as f:
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

    # Unique run name (so multiple notebook runs don't collide)
    run_name = f"hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build command
    repo_root = get_repo_root()
    train_cmd = [
        "uv",
        "run",  # Add uv run prefix for proper environment activation
        str(repo_root / "tools" / "train.py"),
        f"run={run_name}",
        f"trainer.curriculum=tmp/{curriculum_name}",
        "wandb=off",
        "device=cpu",
        "trainer.total_timesteps=100000",
        "trainer.batch_size=256",  # Must be divisible by bptt_horizon
        "trainer.minibatch_size=128",  # Adjusted for bptt_horizon=64
        "trainer.num_workers=8",
        "trainer.bptt_horizon=64",  # Longer horizon for better temporal learning
        "trainer.forward_pass_minibatch_target_size=2",
        "trainer.simulation.skip_git_check=true",  # Skip git check to avoid errors in notebooks
        "trainer.checkpoint.checkpoint_interval=500",  # Skip git check to avoid errors in notebooks
        "trainer.simulation.evaluate_interval=500",  # Skip git check to avoid errors in notebooks
        "trainer.simulation.evaluate_remote=false",  # Skip git check to avoid errors in notebooks
        "sim=sim",
        "+train_job.evals.name=hallway",
        "+train_job.evals.num_episodes=1",
        "+train_job.evals.simulations={}",
    ]

    process = subprocess.Popen(
        train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout or []:
        print(line, end="")
    process.wait()

    temp_curriculum_path.unlink(missing_ok=True)
    return (run_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Understanding Training Results

    - **Logs** live in `train_dir/{run_name}/*.log`
    - **Checkpoints** (PyTorch `.pt` files) are in `train_dir/{run_name}/checkpoints/`
      the latest one is the policy we’ll load next.
    - **Replays** (optional) would be in `train_dir/{run_name}/replays/`

    You can inspect the logs or open a checkpoint later to see the learned network weights.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Observing the Trained Agent

    Let’s load the newest checkpoint and watch the trained policy in the same hallway environment.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    observe_button2 = mo.ui.run_button(
        label="Click to observe your trained agent below"
    )
    observe_button2
    return (observe_button2,)


@app.cell
def _(
    OmegaConf,
    Path,
    contextlib,
    display,
    env_dict: "Dict[str, Any]",
    get_policy,
    io,
    mo,
    observe_button2,
    run_name,
    setup_environment,
    time,
    widgets,
):
    mo.stop(not observe_button2.value)
    ckpt_dir = Path("train_dir") / run_name / "checkpoints"
    latest_ckpt = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    print("Loading", latest_ckpt.name)
    auto_cfg = OmegaConf.create(
        {
            "env": env_dict,
            "policy_uri": f"file://{latest_ckpt.absolute()}",
            "renderer_job": {
                "policy_type": "trained",
                "num_steps": 1000,
                "num_agents": 1,
                "sleep_time": 0.04,
            },
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        trained_env, _ = setup_environment(auto_cfg)
        trained_policy = get_policy("trained", trained_env, auto_cfg)
    header2 = widgets.HTML()
    map_box2 = widgets.HTML()
    display(header2, map_box2)
    _obs, _ = trained_env.reset()
    for _step in range(auto_cfg.renderer_job.num_steps * 10):
        _actions = trained_policy.predict(_obs)
        print(_actions)
        _obs, _, _, _, _ = trained_env.step(_actions)
        _agent_obj = next(
            (o for o in trained_env.grid_objects.values() if o.get("agent_id") == 0)
        )
        _inv = {
            trained_env.inventory_item_names[i]: c
            for i, c in _agent_obj.get("inventory", {}).items()
        }
        header2.value = f"<b>Step:</b> {_step + 1}/{auto_cfg.renderer_job.num_steps} <br/> <b>Inventory:</b> {_inv}"
        with contextlib.redirect_stdout(io.StringIO()):
            buf = trained_env.render()
        map_box2.value = f"<pre>{buf}</pre>"
        if auto_cfg.renderer_job.sleep_time:
            time.sleep(auto_cfg.renderer_job.sleep_time)
    trained_env.close()
    return


if __name__ == "__main__":
    app.run()
