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

    # Import MettaScope replay viewer
    try:
        from experiments.notebooks.utils.replays import show_replay

        replay_available = True
    except ImportError:
        replay_available = False
        print("⚠️ MettaScope replay viewer not available")

    from metta.agent.policy_store import PolicyStore

    from metta.common.wandb.wandb_context import WandbConfig
    from metta.rl.policy_management import initialize_policy_for_environment
    import torch

    from tensordict import TensorDict

    import logging
    from metta.tools.train import TrainTool
    from metta.rl.trainer_config import (
        TrainerConfig,
        CheckpointConfig,
        EvaluationConfig,
    )

    from metta.cogworks.curriculum import env_curriculum

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
        num_agents: int = 1
        max_steps: int = 100000
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
            """Wander randomly and get ore if next to a mine."""
            grid_objects = self.env.grid_objects
            agent = next(
                (o for o in grid_objects.values() if o.get("agent_id") == 0), None
            )
            if agent is None:
                return generate_valid_random_actions(self.env, self.num_agents)

            ar, ac = agent["r"], agent["c"]
            agent_ori = int(agent.get("agent:orientation", 0))

            # Check if next to a mine with ore - if so, pick up
            for orient, (dr, dc) in self.ORIENT_TO_DELTA.items():
                tr, tc = ar + dr, ac + dc
                for obj in grid_objects.values():
                    if obj.get("r") == tr and obj.get("c") == tc:
                        obj_type_id = obj.get("type")
                        if obj_type_id is not None and obj_type_id < len(
                            self.env.object_type_names
                        ):
                            obj_type_name = self.env.object_type_names[obj_type_id]
                            if "mine" in obj_type_name:
                                inv = obj.get("inventory", {})
                                total = (
                                    sum(inv.values()) if isinstance(inv, dict) else 0
                                )
                                if total > 0:
                                    # If facing the mine, pick up; otherwise rotate toward it
                                    if orient == agent_ori:
                                        action_type, action_arg = self.pickup_idx, 0
                                    else:
                                        action_type, action_arg = (
                                            self.rotate_idx,
                                            orient,
                                        )
                                    return generate_valid_random_actions(
                                        self.env,
                                        self.num_agents,
                                        force_action_type=action_type,
                                        force_action_arg=action_arg,
                                    )

            # Otherwise, wander randomly
            return generate_valid_random_actions(self.env, self.num_agents)

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
        CheckpointConfig,
        EvaluationConfig,
        MettaGridEnv,
        Path,
        PolicyStore,
        RendererToolConfig,
        TensorDict,
        TrainTool,
        TrainerConfig,
        WandbConfig,
        contextlib,
        datetime,
        display,
        env_curriculum,
        get_policy,
        initialize_policy_for_environment,
        io,
        logging,
        mo,
        np,
        os,
        pd,
        replay_available,
        show_replay,
        time,
        torch,
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
    from metta.mettagrid.mettagrid_config import AgentRewards, InventoryRewards
    import pprint

    # Define simple hallway map as ASCII string
    import textwrap

    hallway_map = textwrap.dedent("""
        ###########
        #@.......m#
        ###########
    """).strip()

    # Start with working arena config for 1 agent, then customize
    env_config = make_arena(num_agents=1, combat=False)

    # Replace with our simple hallway map
    map_data = [list(line) for line in hallway_map.splitlines()]
    env_config.game.map_builder = AsciiMapBuilder.Config(map_data=map_data)

    # Simple customizations
    env_config.game.max_steps = 5000
    env_config.game.obs_width = 11
    env_config.game.obs_height = 11

    # Enable basic movement and item collection - disable combat
    env_config.game.actions.move_cardinal.enabled = True
    env_config.game.actions.rotate.enabled = True
    env_config.game.actions.noop.enabled = True
    env_config.game.actions.move_8way.enabled = False
    env_config.game.actions.move.enabled = False
    env_config.game.actions.change_color.enabled = False
    env_config.game.actions.change_glyph.enabled = False
    env_config.game.actions.swap.enabled = False
    env_config.game.actions.place_box.enabled = False

    # Ensure ore collection gives rewards
    env_config.game.agent.rewards = AgentRewards(
        inventory=InventoryRewards(
            ore_red=0.1,
            ore_red_max=255,
            battery_red=0.8,
            battery_red_max=255,
        ),
    )

    # Set initial resource counts for immediate availability
    for obj_name in ["mine_red", "generator_red"]:
        if obj_name in env_config.game.objects:
            obj_copy = env_config.game.objects[obj_name].model_copy(deep=True)
            obj_copy.initial_resource_count = 10
            env_config.game.objects[obj_name] = obj_copy

    # Create a proper RendererToolConfig for policy creation
    renderer_config = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=1000,
        sleep_time=0.010,
        renderer_type="human",
    )

    # Global configuration flags from old mettagrid.yaml
    env_config.desync_episodes = True  # Changes max_steps for first episode only
    env_config.game.track_movement_metrics = True
    env_config.game.no_agent_interference = False
    env_config.game.recipe_details_obs = False

    # Global observation tokens from old config
    env_config.game.global_obs.episode_completion_pct = True
    env_config.game.global_obs.last_action = True
    env_config.game.global_obs.last_reward = True

    env_config.game.global_obs.visitation_counts = False

    print("✅ Simple hallway environment: start with arena, add custom map")
    print(pprint.pp(env_config, indent=1, width=80))
    return (
        AgentRewards,
        AsciiMapBuilder,
        InventoryRewards,
        env_config,
        make_arena,
        renderer_config,
        textwrap,
    )


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
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                buffer_str = env.render()
            map_box.value = f"<pre>{buffer_str}</pre>"
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
def _(
    CheckpointConfig,
    EvaluationConfig,
    TrainTool,
    TrainerConfig,
    datetime,
    env_config,
    env_curriculum,
    logging,
    mo,
    os,
    train_button,
):
    username = os.environ.get("USER", "metta_user")

    # Unique run name (so multiple notebook runs don't collide)
    run_name = (
        f"{username}.hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"🚀 Starting training run: {run_name}")

    def train_agent():
        # Create a simple curriculum with our hallway environment
        curriculum = env_curriculum(env_config)

        # Create trainer configuration with Mac-optimized settings
        # Batch sizes optimized for Mac CPU/Metal - larger than demo but reasonable for local machine
        trainer_config = TrainerConfig(
            curriculum=curriculum,
            total_timesteps=2000000,  # Small demo run
            # total_timesteps=1000,  # DEBUG run
            batch_size=65536,  # Increased from 256, reduced from default 524288 for Mac
            minibatch_size=512,  # Increased from 256, reduced from default 16384 for Mac
            rollout_workers=14,  # Single worker for Mac
            forward_pass_minibatch_target_size=512,  # Increased from 2 - better GPU utilization
            # Adjusted learning rate for smaller batch size (scaled down from default 0.000457)
            # Using sqrt(2048/524288) ≈ 0.0625 scaling factor
            # optimizer={
            #    "learning_rate": 0.00003
            # },  # Reduced from default for smaller batch
            checkpoint=CheckpointConfig(
                checkpoint_interval=10,  # Checkpoint every n epochs
                wandb_checkpoint_interval=10,
            ),
            # Enable replay generation for MettaScope visualization
            evaluation=EvaluationConfig(
                evaluate_interval=10,  # Generate replays every 50 steps (matches checkpoint interval)
                evaluate_remote=False,  # Run locally or you'll have to wait
                evaluate_local=True,
                replay_dir=f"s3://softmax-public/replays/{run_name}",  # Store replays on S3
            ),
        )

        # Create and configure the training tool
        train_tool = TrainTool(
            trainer=trainer_config,
            # wandb=WandbConfigOff(),  # Disable wandb for simplicity
            run=run_name,
            run_dir=f"train_dir/{run_name}",
            disable_macbook_optimize=True,
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

    mo.stop(not train_button.value)
    train_agent()
    return run_name, username


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
    PolicyStore,
    TensorDict,
    WandbConfig,
    contextlib,
    display,
    env_config,
    eval_trained_button,
    initialize_policy_for_environment,
    io,
    mo,
    np,
    pd,
    renderer_config,
    run_name,
    time,
    torch,
    widgets,
):
    # Load trained policy using repo's PolicyStore approach (like tools/sim.py)
    mo.stop(not eval_trained_button.value or not run_name)

    def evaluate_agent():
        # Find the latest checkpoint
        ckpt_dir = Path("train_dir") / run_name / "checkpoints"
        latest_ckpt = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)

        print(f"Evaluating checkpoint: {latest_ckpt.name}")

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

                td = TensorDict({"env_obs": obs_tensor}, batch_size=obs_tensor.shape[0])
                trained_policy(td)
                _actions = td["actions"].cpu().numpy()

                _obs, _, _, _, _ = eval_env.step(_actions)
                obs_tensor = torch.as_tensor(_obs, device=torch.device("cpu"))

                # Update display every few steps to show animation
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
                    f"<br />"
                    f"<b>Ore collected:</b> {_inv.get('ore_red', 0)}"
                )
                with contextlib.redirect_stdout(io.StringIO()) as buffer:
                    buffer_str = eval_env.render()
                map_box.value = f"<pre>{buffer_str}</pre>"
                time.sleep(renderer_config.sleep_time)  # Small delay for animation

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

    evaluate_agent()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. Visualizing Agent Behavior with MettaScope

    Now let's view a replay of our trained agent using MettaScope - an interactive visualization tool that shows:

    - **Agent movement and behavior** over time
    - **Resource collection events** as they happen
    - **Environment state changes** step by step
    - **Interactive timeline** to scrub through the episode

    MettaScope provides a much richer view than the ASCII rendering, showing the full strategic behavior of our trained agent.

    **Note**: Replays are generated during training evaluation. If no replays appear, the training may not have completed the evaluation phase yet.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    replay_button = mo.ui.run_button(label="Click to view MettaScope replay")
    replay_button
    return (replay_button,)


@app.cell
def _(mo, replay_available, replay_button, run_name, show_replay):
    mo.stop(not replay_button.value or not run_name)

    if not replay_available:
        print("❌ MettaScope replay viewer is not available")
        print("Make sure you're running from the experiments/marimo directory")
    else:
        print(f"🎬 Loading MettaScope replay for training run: {run_name}")
        print("This will display an interactive visualization of the trained agent...")

        try:
            # Show the latest replay from the training run
            show_replay(run_name, step="last", width=950, height=400, autoplay=True)
        except Exception as e:
            print(f"❌ Error loading replay: {e}")
            print("This could mean:")
            print("- Training hasn't generated replays yet (evaluation incomplete)")
            print("- Run not found in W&B (check run name)")
            print("- Network connectivity issues")
            print(
                f"\nReplays are stored on S3 at: s3://softmax-public/replays/{run_name}/"
            )
    return


@app.cell
def _(mo, run_name, traceback):
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

            return mo.md(obj.to_html(height=height))
        except wandb.Error:
            traceback.print_exc()
            return mo.md(
                f"Path {path!r} does not refer to a W&B object you can access."
            )

    _display_by_wandb_path(f"metta-research/metta/runs/{run_name}", height=580)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Let's run a new example with a new map

        ###########  
        #R...@...m#  
        ###########
    """
    )
    return


@app.cell
def _(
    AgentRewards,
    AsciiMapBuilder,
    InventoryRewards,
    RendererToolConfig,
    make_arena,
    textwrap,
):
    hallway_map2 = textwrap.dedent("""
        ###########
        #R...@...m#
        ###########
    """).strip()

    # Start with working arena config for 1 agent, then customize
    env_config2 = make_arena(num_agents=1)

    # Replace with our simple hallway map
    map_data2 = [list(line) for line in hallway_map2.splitlines()]
    env_config2.game.map_builder = AsciiMapBuilder.Config(map_data=map_data2)

    # Simple customizations
    env_config2.game.max_steps = 5000
    env_config2.game.obs_width = 11
    env_config2.game.obs_height = 11

    # Enable basic movement and item collection - disable combat
    env_config2.game.actions.move_cardinal.enabled = True
    env_config2.game.actions.rotate.enabled = True
    env_config2.game.actions.noop.enabled = True
    env_config2.game.actions.move_8way.enabled = False
    env_config2.game.actions.attack.enabled = False
    env_config2.game.actions.put_items.enabled = True
    env_config2.game.actions.change_color.enabled = False
    env_config2.game.actions.change_glyph.enabled = False
    env_config2.game.actions.swap.enabled = False
    env_config2.game.actions.place_box.enabled = False

    # Ensure ore collection gives rewards
    env_config2.game.agent.rewards = AgentRewards(
        inventory=InventoryRewards(
            ore_red=1.0,
            battery_red=1.0,
        ),
    )

    renderer_config2 = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=3000,
        sleep_time=0.005,
        renderer_type="human",
    )
    return env_config2, renderer_config2


@app.cell
def _(mo):
    observe_button2 = mo.ui.run_button(label="Click to run observation below")
    observe_button2
    return (observe_button2,)


@app.cell
def _(
    MettaGridEnv,
    contextlib,
    display,
    env_config2,
    get_policy,
    io,
    mo,
    observe_button2,
    renderer_config2,
    time,
    widgets,
):
    mo.stop(not observe_button2.value)

    def observe_agent2():
        # Create environment with proper EnvConfig
        env = MettaGridEnv(env_config2, render_mode="human")
        policy = get_policy(renderer_config2.policy_type, env, renderer_config2)

        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)
        _obs, info = env.reset()

        # steps = renderer_config.num_steps
        steps = renderer_config2.num_steps
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
            header.value = "<br />".join(
                [
                    f"<b>Step:</b> {_step + 1}/{steps}",
                    f"<b>Inventory:</b> ore={_inv.get('ore_red', 0)} batteries={_inv.get('battery_red', 0)}",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                buffer_str = env.render()
            map_box.value = f"<pre>{buffer_str}</pre>"
            time.sleep(renderer_config2.sleep_time)
        env.close()

    observe_agent2()
    return


@app.cell
def _(mo):
    train_button2 = mo.ui.run_button(label="Click to run training below")
    train_button2
    return (train_button2,)


@app.cell
def _(
    CheckpointConfig,
    EvaluationConfig,
    TrainTool,
    TrainerConfig,
    datetime,
    env_config2,
    env_curriculum,
    logging,
    mo,
    train_button2,
    username,
):
    mo.stop(not train_button2.value)

    def train_agent2():
        # Create a simple curriculum with our hallway environment
        curriculum = env_curriculum(env_config2)

        run_name2 = f"{username}.hello_world_train.generator.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create trainer configuration with Mac-optimized settings
        # Batch sizes optimized for Mac CPU/Metal - larger than first demo but still reasonable
        trainer_config = TrainerConfig(
            curriculum=curriculum,
            total_timesteps=2000000,  # Small demo run
            # total_timesteps=1000,  # DEBUG run
            batch_size=65536,  # Increased from 256, reduced from default 524288 for Mac
            minibatch_size=512,  # Increased from 256, reduced from default 16384 for Mac
            rollout_workers=14,  # Single worker for Mac
            forward_pass_minibatch_target_size=512,  # Increased from 2 - better GPU utilization
            # Adjusted learning rate for smaller batch size (scaled down from default 0.000457)
            # Using sqrt(4096/524288) ≈ 0.088 scaling factor
            optimizer={
                "learning_rate": 0.00004
            },  # Reduced from default for smaller batch
            checkpoint=CheckpointConfig(
                checkpoint_interval=10,  # Checkpoint every n epochs
                wandb_checkpoint_interval=10,
            ),
            # Enable replay generation for MettaScope visualization
            evaluation=EvaluationConfig(
                evaluate_interval=10,  # Generate replays every 50 steps (matches checkpoint interval)
                evaluate_remote=False,  # Run locally or you'll have to wait
                evaluate_local=True,
                replay_dir=f"s3://softmax-public/replays/{run_name2}",  # Store replays on S3
            ),
        )

        # Create and configure the training tool
        train_tool = TrainTool(
            trainer=trainer_config,
            # wandb=WandbConfigOff(),  # Disable wandb for simplicity
            run=run_name2,
            run_dir=f"train_dir/{run_name2}",
            disable_macbook_optimize=True,
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

        return run_name2

    run_name2 = train_agent2()
    return (run_name2,)


@app.cell
def _(mo):
    eval_trained_button2 = mo.ui.run_button(label="Click to evaluate trained agent")
    eval_trained_button2
    return (eval_trained_button2,)


@app.cell
def _(
    EVAL_EPISODES,
    MettaGridEnv,
    Path,
    PolicyStore,
    TensorDict,
    WandbConfig,
    contextlib,
    display,
    env_config2,
    eval_trained_button2,
    initialize_policy_for_environment,
    io,
    mo,
    np,
    pd,
    renderer_config2,
    run_name2,
    time,
    torch,
    widgets,
):
    mo.stop(not eval_trained_button2.value or not run_name2)

    def evaluate_agent2():
        # Find the latest checkpoint
        ckpt_dir = Path("train_dir") / run_name2 / "checkpoints"
        latest_ckpt = max(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)

        print(f"Evaluating checkpoint: {latest_ckpt.name}")

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
            eval_env = MettaGridEnv(env_config2, render_mode="human")

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
        trained_scores_ore: list[int] = []
        trained_scores_batteries: list[int] = []

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
                td = TensorDict({"env_obs": obs_tensor}, batch_size=obs_tensor.shape[0])
                trained_policy(td)
                _actions = td["actions"].cpu().numpy()

                _obs, _, _, _, _ = eval_env.step(_actions)
                obs_tensor = torch.as_tensor(_obs, device=torch.device("cpu"))

                # Update display every few steps to show animation
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
                    f"<br />"
                    f"<b>Ore collected:</b> {_inv.get('ore_red', 0)}"
                    f"<br />"
                    f"<b>Batteries collected:</b> {_inv.get('battery_red', 0)}"
                )
                with contextlib.redirect_stdout(io.StringIO()) as buffer:
                    buffer_str = eval_env.render()
                map_box.value = f"<pre>{buffer_str}</pre>"
                time.sleep(renderer_config2.sleep_time)  # Small delay for animation

            # Final inventory count for this episode
            _agent_obj = next(
                (o for o in eval_env.grid_objects.values() if o.get("agent_id") == 0)
            )
            _inv = {
                eval_env.inventory_item_names[idx]: cnt
                for idx, cnt in _agent_obj.get("inventory", {}).items()
            }
            inv_count_ore = int(_inv.get("ore_red", 0))
            inv_count_batteries = int(_inv.get("battery_red", 0))
            trained_scores_ore.append(inv_count_ore)
            trained_scores_batteries.append(inv_count_batteries)

        eval_env.close()

        # Calculate and display results
        mean_score_ore = np.mean(trained_scores_ore)
        mean_score_batteries = np.mean(trained_scores_batteries)
        std_score_ore = np.std(trained_scores_ore)
        std_score_batteries = np.std(trained_scores_batteries)
        running_avg_ore = pd.Series(trained_scores_ore).expanding().mean()
        running_avg_batteries = pd.Series(trained_scores_batteries).expanding().mean()

        # Show final results
        header.value = f"<b>✅ Evaluation Complete!</b>"
        map_box.value = f"""<pre>
    🏆 TRAINED AGENT RESULTS 🏆

    Episodes: {EVAL_EPISODES}
    Average Score: {mean_score_ore:.2f} ± {std_score_ore:.2f} ore collected
    Average Score: {mean_score_batteries:.2f} ± {std_score_batteries:.2f} batteries collected
    Best Episode: {max(trained_scores_ore)} ore, {max(trained_scores_batteries)} batteries
    Worst Episode: {min(trained_scores_ore)} ore, {min(trained_scores_batteries)} batteries

    Individual Episode Scores: {trained_scores_ore} ore, {trained_scores_batteries} batteries

    Compare this to the opportunistic baseline from earlier!
        </pre>"""

        display(
            pd.DataFrame(
                {
                    "episode": list(range(1, EVAL_EPISODES + 1)),
                    "ore_red": trained_scores_ore,
                    "battery_red": trained_scores_batteries,
                    "running_avg_ore": running_avg_ore,
                    "running_avg_batteries": running_avg_batteries,
                }
            )
        )

        print(
            f"\n🎯 Trained agent performance: {mean_score_ore:.2f} ± {std_score_ore:.2f} ore collected"
        )
        print(
            f"\n🎯 Trained agent performance: {mean_score_batteries:.2f} ± {std_score_batteries:.2f} batteries collected"
        )
        print(f"📊 Compare with opportunistic baseline from earlier evaluation!")

    evaluate_agent2()
    return


if __name__ == "__main__":
    app.run()
