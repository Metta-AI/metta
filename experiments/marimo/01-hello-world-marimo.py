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
    from metta.common.util.fs import get_repo_root
    import anywidget
    import traitlets
    from IPython.display import display
    from metta.mettagrid import MettaGridEnv

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

    # Policy implementations (replacing the deprecated tools/renderer.py)
    from metta.common.config import Config
    from metta.mettagrid.util.actions import generate_valid_random_actions
    from typing import Protocol, List
    import numpy as np

    class RendererToolConfig(Config):
        policy_type: str = "random"
        policy_uri: str | None = None
        num_steps: int = 50000
        num_agents: int = 2
        max_steps: int = 10000
        sleep_time: float = 0.0
        renderer_type: str = "human"

    class Policy(Protocol):
        """Protocol for policy classes."""

        def predict(self, obs: np.ndarray) -> np.ndarray:
            """Predict actions given observations."""
            ...

    class BasePolicy:
        """Base class for all policies."""

        def __init__(self, env: MettaGridEnv) -> None:
            self.env = env
            self.num_agents = env.num_agents
            self.action_space = env.action_space
            self.single_action_space = env.single_action_space

        def predict(self, obs: np.ndarray) -> np.ndarray:
            """Predict actions given observations."""
            raise NotImplementedError

    class RandomPolicy(BasePolicy):
        """Simple random policy using valid action generation."""

        def predict(self, obs: np.ndarray) -> np.ndarray:
            """Return valid random actions for all agents."""
            return generate_valid_random_actions(self.env, self.num_agents)

    class OpportunisticPolicy(BasePolicy):
        """Wander; pick up if front, else rotate toward adjacent resource; else roam."""

        ORIENT_TO_DELTA = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
        }
        DELTA_TO_ORIENT = {v: k for k, v in ORIENT_TO_DELTA.items()}

        def __init__(self, env: MettaGridEnv) -> None:
            super().__init__(env)
            # Movement options
            self.cardinal_directions: List[int] = [1, 3, 5, 7]
            self.rotation_orientations: List[int] = [0, 1, 2, 3]
            self._initialize_action_indices()

        def _initialize_action_indices(self) -> None:
            """Determine indices of move/rotate/pickup actions for this env."""
            try:
                action_names: List[str] = self.env.action_names
                self.move_idx: int = (
                    action_names.index("move_cardinal")
                    if "move_cardinal" in action_names
                    else 0
                )
                self.rotate_idx: int = (
                    action_names.index("rotate") if "rotate" in action_names else 1
                )
                # Prefer modern name; accept legacy alias
                if "get_items" in action_names:
                    self.pickup_idx = action_names.index("get_items")
                elif "pickup" in action_names:
                    self.pickup_idx = action_names.index("pickup")
                else:
                    self.pickup_idx = 2
            except (AttributeError, ValueError):
                # Fallback defaults
                self.move_idx = 0
                self.rotate_idx = 1
                self.pickup_idx = 2

        def predict(self, obs: np.ndarray) -> np.ndarray:
            """Decide a step action following opportunistic rules."""
            grid_objects = self.env.grid_objects
            agent = next(
                (o for o in grid_objects.values() if o.get("agent_id") == 0), None
            )
            if agent is None:
                return generate_valid_random_actions(self.env, self.num_agents)

            ar, ac = agent["r"], agent["c"]
            agent_ori = int(agent.get("agent:orientation", 0))

            # Adjacent resource? pick up if front, else rotate toward it
            for orient, (dr, dc) in self.ORIENT_TO_DELTA.items():
                tr, tc = ar + dr, ac + dc
                for obj in grid_objects.values():
                    if obj.get("r") == tr and obj.get("c") == tc:
                        base = self.env.object_type_names[obj["type"]].split(".")[0]
                        if base.startswith(("mine", "generator", "converter")):
                            inv = obj.get("inventory", {})
                            total = sum(inv.values()) if isinstance(inv, dict) else 0
                            if total > 0:
                                action_type, action_arg = (
                                    (self.pickup_idx, 0)
                                    if orient == agent_ori
                                    else (self.rotate_idx, orient)
                                )
                                return generate_valid_random_actions(
                                    self.env,
                                    self.num_agents,
                                    force_action_type=action_type,
                                    force_action_arg=action_arg,
                                )

            # Roam
            occupied = {
                (o["r"], o["c"]) for o in grid_objects.values() if o.get("layer") == 1
            }
            candidates = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            moves = [
                (dr, dc) for dr, dc in candidates if (ar + dr, ac + dc) not in occupied
            ]
            if moves:
                dr, dc = moves[np.random.randint(len(moves))]
                desired_ori = self.DELTA_TO_ORIENT[(dr, dc)]
                action_type, action_arg = (
                    (self.rotate_idx, desired_ori)
                    if agent_ori != desired_ori
                    else (self.move_idx, 0)
                )
            else:
                action_type, action_arg = (
                    self.rotate_idx,
                    int(np.random.choice(self.rotation_orientations)),
                )

            return generate_valid_random_actions(
                self.env,
                self.num_agents,
                force_action_type=action_type,
                force_action_arg=action_arg,
            )

    def get_policy(
        policy_type: str, env: MettaGridEnv, cfg: RendererToolConfig
    ) -> Policy:
        """Get a policy based on the specified type."""
        if policy_type == "random":
            return RandomPolicy(env)
        elif policy_type == "opportunistic":
            return OpportunisticPolicy(env)
        else:
            raise Exception("Unknown policy type")

    print("Setup done")
    return (
        MettaGridEnv,
        Path,
        RendererToolConfig,
        contextlib,
        datetime,
        display,
        get_policy,
        io,
        mo,
        np,
        pd,
        time,
        widgets,
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
def _(RendererToolConfig):
    # Simple approach: use the built-in arena and add a custom map - just like the demos do
    from metta.mettagrid.config.envs import make_arena
    from metta.mettagrid.map_builder.ascii import AsciiMapBuilder

    # Define simple hallway map as ASCII string
    import textwrap

    hallway_map = textwrap.dedent("""
        ###########
        #@.......m#
        ###########
    """).strip()

    # Start with working arena config for 1 agent, then customize
    env_config = make_arena(num_agents=1)

    # Replace with our simple hallway map
    map_data = [list(line) for line in hallway_map.splitlines()]
    env_config.game.map_builder = AsciiMapBuilder.Config(map_data=map_data)

    # Simple customizations
    env_config.game.max_steps = 1000
    env_config.game.obs_width = 11
    env_config.game.obs_height = 11

    # Enable basic movement and item collection - disable combat
    env_config.game.actions.move_cardinal.enabled = True
    env_config.game.actions.rotate.enabled = True
    env_config.game.actions.noop.enabled = True
    env_config.game.actions.move_8way.enabled = False
    env_config.game.actions.attack.enabled = False
    env_config.game.actions.put_items.enabled = False
    env_config.game.actions.change_color.enabled = False
    env_config.game.actions.change_glyph.enabled = False
    env_config.game.actions.swap.enabled = False
    env_config.game.actions.place_box.enabled = False

    # Ensure ore collection gives rewards
    env_config.game.agent.rewards.inventory.ore_red = 1.0

    # Create a proper RendererToolConfig for policy creation
    renderer_config = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=100,
        sleep_time=0.010,
        renderer_type="human",
    )

    print("✅ Simple hallway environment: start with arena, add custom map")
    return env_config, renderer_config


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
    MettaGridEnv,
    contextlib,
    display,
    env_config,
    get_policy,
    io,
    mo,
    observe_button,
    renderer_config,
    time,
    widgets,
):
    mo.stop(not observe_button.value)

    import re

    def _():
        # Create environment with proper EnvConfig
        env = MettaGridEnv(env_config, render_mode="human")
        policy = get_policy(renderer_config.policy_type, env, renderer_config)

        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)
        _obs, info = env.reset()

        # steps = renderer_config.num_steps
        steps = renderer_config.num_steps
        for _step in range(steps):
            _actions = policy.predict(_obs)
            _obs, rewards, terminals, truncations, info = env.step(_actions)
            _agent_obj = next(
                (o for o in env.grid_objects.values() if o.get("agent_id") == 0)
            )
            _inv = {
                env.inventory_item_names[idx]: count
                for idx, count in _agent_obj.get("inventory", {}).items()
            }
            header.value = f"<b>Step:</b> {_step + 1}/{steps} <br/> <b>Inventory:</b> {_inv.get('ore_red', 0)}"
            # Get rendered output directly
            buffer_str = env.render()
            # Clean ANSI escape codes for web display
            clean_buffer = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", buffer_str)
            map_box.value = f"<pre>{clean_buffer}</pre>"
            if renderer_config.sleep_time:
                time.sleep(renderer_config.sleep_time)
        env.close()

    _()
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
    MettaGridEnv,
    contextlib,
    display,
    env_config,
    eval_button,
    get_policy,
    io,
    mo,
    np,
    pd,
    renderer_config,
):
    EVAL_EPISODES = 10
    scores: list[int] = []

    mo.stop(not eval_button.value)

    with contextlib.redirect_stdout(io.StringIO()):
        # Create evaluation environment with our simple config
        eval_env = MettaGridEnv(env_config, render_mode="human")
        eval_policy = get_policy(renderer_config.policy_type, eval_env, renderer_config)

    for ep in range(1, EVAL_EPISODES + 1):
        _obs, _ = eval_env.reset()
        inv_count = 0
        for _step in range(renderer_config.num_steps):
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

    mean_score = np.mean(scores)
    std_score = np.std(scores)
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
    print(
        f"Opportunistic agent baseline: {mean_score:.2f} ± {std_score:.2f} ore collected"
    )
    return (EVAL_EPISODES,)


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
def _(datetime, env_config, mo, train_button):
    mo.stop(not train_button.value)

    # Import training modules
    import logging
    from metta.tools.train import TrainTool
    from metta.rl.trainer_config import (
        TrainerConfig,
        CheckpointConfig,
        EvaluationConfig,
    )

    # from metta.common.wandb.wandb_context import WandbConfigOff
    from metta.cogworks.curriculum import env_curriculum

    # Unique run name (so multiple notebook runs don't collide)
    run_name = f"hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"🚀 Starting training run: {run_name}")

    # Create a simple curriculum with our hallway environment
    curriculum = env_curriculum(env_config)

    # Create trainer configuration with small settings for demo
    trainer_config = TrainerConfig(
        curriculum=curriculum,
        total_timesteps=20000,  # Small demo run
        # total_timesteps=1000,  # DEBUG run
        batch_size=256,
        minibatch_size=256,
        rollout_workers=14,  # Correct field name
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,  # Checkpoint every 50 steps for demo
            wandb_checkpoint_interval=50,
        ),
        # Disable evaluations for simplicity
        evaluation=EvaluationConfig(
            evaluate_interval=0,  # Disable evaluations
            evaluate_remote=False,
            simulations=[],  # Empty list instead of None
        ),
    )

    # Create and configure the training tool
    train_tool = TrainTool(
        trainer=trainer_config,
        # wandb=WandbConfigOff(),  # Disable wandb for simplicity
        run=run_name,
        run_dir=f"train_dir/{run_name}",
    )

    # Set up logging to capture output
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        print("🏋️ Training started...")
        result = train_tool.invoke()  # Use invoke() method instead of run()
        print(f"✅ Training completed successfully! Result: {result}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    return run_name, traceback


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
    ## 7. Evaluating the Trained Agent

    Now let's evaluate our trained agent using the same evaluation infrastructure that `tools/sim.py` uses internally. This will run the trained policy on multiple episodes of the hallway environment and compare its performance to the opportunistic baseline.

    **Note**: The visual observation of trained agents is currently not implemented in the renderer (it shows "TODO: this feature got broken after pydantic config migration"), but the quantitative evaluation below works perfectly and shows the improvement from training.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    eval_trained_button = mo.ui.run_button(label="Click to evaluate trained agent")
    eval_trained_button
    return (eval_trained_button,)


@app.cell
def _(
    EVAL_EPISODES,
    MettaGridEnv,
    Path,
    contextlib,
    display,
    env_config,
    eval_trained_button,
    io,
    mo,
    np,
    pd,
    run_name,
    time,
    widgets,
):
    mo.stop(not eval_trained_button.value)

    def _():
        # Find the latest checkpoint
        ckpt_dir = Path("train_dir") / run_name / "checkpoints"
        latest_ckpt = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)

        print(f"Evaluating checkpoint: {latest_ckpt.name}")

        # Load trained policy using repo's PolicyStore approach (like tools/sim.py)
        from metta.agent.policy_store import PolicyStore

        from metta.common.wandb.wandb_context import WandbConfig
        from metta.rl.policy_management import initialize_policy_for_environment
        import torch

        # Create policy store (same as tools/sim.py:65-70)
        policy_store = PolicyStore.create(
            device="cpu",
            wandb_config=WandbConfig.Off(),
            data_dir="train_dir",
            wandb_run=None,
        )

        # Get policy record (same as tools/sim.py:76-82)
        policy_uri = f"file://{latest_ckpt.parent.absolute()}"
        policy_records = policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type="latest",
            n=1,
            metric="score",
        )

        if not policy_records:
            raise Exception("No policy records found")

        policy_record = policy_records[0]
        print(f"✅ Successfully loaded policy: {policy_record.run_name}")

        # Create evaluation environment
        with contextlib.redirect_stdout(io.StringIO()):
            eval_env = MettaGridEnv(env_config, render_mode="human")

        # Initialize policy for environment (same as simulation.py:133-138)
        initialize_policy_for_environment(
            policy_record=policy_record,
            metta_grid_env=eval_env,
            device=torch.device("cpu"),
            restore_feature_mapping=True,
        )

        # Get the trained policy from the policy record
        trained_policy = policy_record.policy

        # Run animated evaluation with the trained policy
        trained_scores: list[int] = []

        # Create header and display widgets for animation
        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)

        print(f"🎯 Running {EVAL_EPISODES} episodes with animated evaluation...")

        for ep in range(1, EVAL_EPISODES + 1):
            header.value = (
                f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Evaluating trained agent..."
            )

            _obs, _ = eval_env.reset()
            # Convert obs to tensor format for policy
            obs_tensor = torch.as_tensor(_obs, device=torch.device("cpu"))

            steps = 1000
            for _step in range(steps):  # Same number of steps as opportunistic
                # Use TensorDict format for trained policy (same as simulation.py:272-275)
                from tensordict import TensorDict

                td = TensorDict({"env_obs": obs_tensor}, batch_size=obs_tensor.shape[0])
                trained_policy(td)
                _actions = td["actions"].cpu().numpy()

                _obs, _, _, _, _ = eval_env.step(_actions)
                obs_tensor = torch.as_tensor(_obs, device=torch.device("cpu"))

                # Update display every few steps to show animation
                if _step % 3 == 0:  # Update every 3 steps for smooth animation
                    _agent_obj = next(
                        (
                            o
                            for o in eval_env.grid_objects.values()
                            if o.get("agent_id") == 0
                        )
                    )
                    _inv = {
                        eval_env.inventory_item_names[idx]: cnt
                        for idx, cnt in _agent_obj.get("inventory", {}).items()
                    }
                    header.value = (
                        f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Step {_step + 1}/{steps} - "
                        f"<b>Ore collected:</b> {_inv.get('ore_red', 0)}"
                    )
                    # Get rendered output directly
                    buffer_str = eval_env.render()
                    # Clean ANSI escape codes for web display
                    import re

                    clean_buffer = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", buffer_str)
                    map_box.value = f"<pre>{clean_buffer}</pre>"
                    time.sleep(0.02)  # Small delay for animation

            # Final inventory count for this episode
            _agent_obj = next(
                (o for o in eval_env.grid_objects.values() if o.get("agent_id") == 0)
            )
            _inv = {
                eval_env.inventory_item_names[idx]: cnt
                for idx, cnt in _agent_obj.get("inventory", {}).items()
            }
            inv_count = int(_inv.get("ore_red", 0))
            trained_scores.append(inv_count)

        eval_env.close()

        # Calculate and display results
        mean_score = np.mean(trained_scores)
        std_score = np.std(trained_scores)
        running_avg = pd.Series(trained_scores).expanding().mean()

        # Show final results
        header.value = f"<b>✅ Evaluation Complete!</b>"
        map_box.value = f"""<pre>
    🏆 TRAINED AGENT RESULTS 🏆

    Episodes: {EVAL_EPISODES}
    Average Score: {mean_score:.2f} ± {std_score:.2f} ore collected
    Best Episode: {max(trained_scores)} ore
    Worst Episode: {min(trained_scores)} ore

    Individual Episode Scores: {trained_scores}

    Compare this to the opportunistic baseline from earlier!
        </pre>"""

        display(
            pd.DataFrame(
                {
                    "episode": list(range(1, EVAL_EPISODES + 1)),
                    "ore_red": trained_scores,
                    "running_avg": running_avg,
                }
            )
        )

        print(
            f"\n🎯 Trained agent performance: {mean_score:.2f} ± {std_score:.2f} ore collected"
        )
        print(f"📊 Compare with opportunistic baseline from earlier evaluation!")

    _()
    return


@app.cell
def _(traceback):
    import wandb
    import IPython

    def _display_by_wandb_path(path: str, *, height: int) -> None:
        """Display a wandb object (usually in an iframe) given its URI.

        Args:
            path: A path to a run, sweep, project, report, etc.
            height: Height of the iframe in pixels.
        """
        api = wandb.Api()

        try:
            obj = api.from_path(path)

            IPython.display.display_html(
                obj.to_html(height=height),
                raw=True,
            )
        except wandb.Error:
            traceback.print_exc()
            IPython.display.display_html(
                f"Path {path!r} does not refer to a W&B object you can access.",
                raw=True,
            )

    _display_by_wandb_path("metta-research/metta", height=600)
    return


if __name__ == "__main__":
    app.run()
