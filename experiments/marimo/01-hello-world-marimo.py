import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="Hello metta-ai")


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

    That's done above in the setup cell.
    """
    )
    return


@app.cell
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
    from contextlib import contextmanager
    import os, json, subprocess, tempfile, yaml
    from datetime import datetime
    import multiprocessing
    import threading
    import traceback

    import numpy as np  # used later
    import pandas as pd
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from typing import Any, Dict  # type: ignore
    from metta.common.util.fs import get_repo_root
    import anywidget
    import traitlets
    from IPython.display import display
    from mettagrid import MettaGridEnv

    # Import MettaScope replay viewer
    try:
        from experiments.notebooks.utils.replays import show_replay

        replay_available = True
    except ImportError:
        replay_available = False
        print("⚠️ MettaScope replay viewer not available")

    from metta.rl.checkpoint_manager import CheckpointManager

    from metta.common.wandb.context import WandbConfig
    import wandb
    import torch

    from tensordict import TensorDict

    import logging
    from metta.tools.train import TrainTool
    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig

    from metta.cogworks.curriculum import (
        env_curriculum,
        CurriculumConfig,
        SingleTaskGenerator,
    )

    # Additional imports for cells
    from mettagrid.builder.envs import make_arena
    from mettagrid.map_builder.ascii import AsciiMapBuilder
    from mettagrid.config.mettagrid_config import (
        AgentRewards,
    )
    from mettagrid.config import Config
    from mettagrid.test_support.actions import generate_valid_random_actions
    from metta.sim.simulation_config import SimulationConfig
    from metta.agent.utils import obs_to_td
    import pprint
    import textwrap
    import signal

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

    # Policy implementations
    from typing import Protocol, List

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

            # Check agent's ore inventory first
            agent_inventory = agent.get("inventory", {})
            agent_ore_count = agent_inventory.get(0, 0)  # ore_red is typically index 0
            max_ore_limit = 10  # Match the resource limit we set

            # Check if next to a mine with ore - only pick up if not at max capacity
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
                                    # If at max ore capacity, move randomly instead of getting stuck
                                    if agent_ore_count >= max_ore_limit:
                                        break  # Skip mine interaction, go to random movement

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

    def get_policy(policy_type: str, env: MettaGridEnv) -> Policy:
        """Get a policy based on the specified type."""
        if policy_type == "random":
            return RandomPolicy(env)
        elif policy_type == "opportunistic":
            return OpportunisticPolicy(env)
        else:
            raise Exception("Unknown policy type")

    @contextmanager
    def cancellable_context():
        """Base context manager for clean cancellation with signal handling"""
        # Only handle signals if we're in the main thread
        # Marimo and other interactive environments run code in separate threads
        is_main_thread = threading.current_thread() is threading.main_thread()

        if is_main_thread:
            original_handler = signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            yield
        except KeyboardInterrupt:
            print("Operation interrupted by user")
            sys.exit(0)
        finally:
            # Always restore the original signal handler if we changed it
            if is_main_thread:
                signal.signal(signal.SIGINT, original_handler)

    @contextmanager
    def training_context():
        """Context manager for training with cleanup of ML resources"""
        with cancellable_context():
            try:
                yield
            except KeyboardInterrupt:
                print("Training interrupted, cleaning up ML resources...")
                # Cleanup training-specific resources
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("✓ GPU memory cleared")
                except Exception as e:
                    print(f"⚠️ GPU cleanup failed: {e}")

                try:
                    if wandb.run is not None:
                        wandb.finish()
                        print("✓ W&B run finished")
                except Exception as e:
                    print(f"⚠️ W&B cleanup failed: {e}")

                print("Training cleanup completed")
                raise  # Re-raise to let cancellable_context handle the exit

    @contextmanager
    def simulation_context(env: MettaGridEnv):
        """Context manager for simulation with cleanup of environment resources"""
        with cancellable_context():
            try:
                yield
            except KeyboardInterrupt:
                print("Simulation interrupted, cleaning up environment resources...")
                # Cleanup simulation-specific resources
                try:
                    # Environment cleanup would go here
                    # e.g., env.close(), release file handles, etc.
                    env.close()
                    print("✓ Environment resources cleaned")
                except Exception as e:
                    print(f"⚠️ Environment cleanup failed: {e}")

                print("Simulation cleanup completed")
                raise  # Re-raise to let cancellable_context handle the exit

    print("Setup done")
    return (
        AgentRewards,
        AsciiMapBuilder,
        Config,
        EvaluatorConfig,
        MettaGridEnv,
        OpportunisticPolicy,
        Path,
        CheckpointManager,
        RendererToolConfig,
        SimulationConfig,
        TensorDict,
        TrainTool,
        TrainerConfig,
        WandbConfig,
        contextlib,
        datetime,
        display,
        env_curriculum,
        generate_valid_random_actions,
        get_repo_root,
        io,
        logging,
        make_arena,
        mo,
        multiprocessing,
        np,
        obs_to_td,
        os,
        threading,
        pd,
        pprint,
        show_replay,
        signal,
        simulation_context,
        textwrap,
        time,
        torch,
        traceback,
        training_context,
        wandb,
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
def _(
    RendererToolConfig,
    make_arena,
    AsciiMapBuilder,
    AgentRewards,
    pprint,
    textwrap,
):
    hallway_map = textwrap.dedent("""
        ###########
        #@.......m#
        ###########
    """).strip()

    # Start with working arena config for 1 agent, then customize
    mg_config = make_arena(num_agents=1, combat=False)

    # Replace with our simple hallway map
    map_data = [list(line) for line in hallway_map.splitlines()]
    mg_config.game.map_builder = AsciiMapBuilder.Config(map_data=map_data)

    # Simple customizations
    mg_config.game.max_steps = 5000
    mg_config.game.obs_width = 11
    mg_config.game.obs_height = 11

    # IMPORTANT: Match the exact training action configuration from config.json
    mg_config.game.actions.move.enabled = True
    mg_config.game.actions.rotate.enabled = True
    mg_config.game.actions.noop.enabled = True  # Training had noop enabled!
    mg_config.game.actions.get_items.enabled = True
    mg_config.game.actions.put_items.enabled = False  # Training had this disabled
    mg_config.game.actions.attack.enabled = True  # Training had attack enabled
    mg_config.game.actions.change_color.enabled = False
    mg_config.game.actions.change_glyph.enabled = False
    mg_config.game.actions.swap.enabled = False

    # IMPORTANT: Match the exact training reward structure from config.json
    mg_config.game.agent.rewards = AgentRewards(
        inventory={
            "ore_red": 0.1,
            "battery_red": 0.8,
        },
        inventory_max={
            "ore_red": 255,
            "battery_red": 255,
        },
    )

    # Use action failure penalty to discourage inefficient actions
    mg_config.game.agent.action_failure_penalty = 0.0  # Match training config

    # Set initial resource counts for immediate availability
    # for obj_name in ["mine_red", "generator_red"]:
    #    if obj_name in mg_config.game.objects:
    #        obj_copy = mg_config.game.objects[obj_name].model_copy(deep=True)
    #        obj_copy.initial_resource_count = 10
    #        mg_config.game.objects[obj_name] = obj_copy

    # Create a proper RendererToolConfig for policy creation
    renderer_config = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=1000,
        sleep_time=0.010,
        renderer_type="human",
    )

    # Global configuration flags from old mettagrid.yaml
    mg_config.desync_episodes = True  # Changes max_steps for first episode only
    mg_config.game.track_movement_metrics = True
    mg_config.game.recipe_details_obs = False

    # Global observation tokens from old config
    mg_config.game.global_obs.episode_completion_pct = True
    mg_config.game.global_obs.last_action = True
    mg_config.game.global_obs.last_reward = True

    mg_config.game.global_obs.visitation_counts = False

    print("✅ Simple hallway environment: start with arena, add custom map")
    return (
        AgentRewards,
        AsciiMapBuilder,
        mg_config,
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
    OpportunisticPolicy,
    contextlib,
    display,
    mg_config,
    io,
    mo,
    observe_button,
    renderer_config,
    simulation_context,
    time,
    widgets,
):
    mo.stop(not observe_button.value)

    def _():
        # Create environment with proper MettaGridConfig
        env = MettaGridEnv(mg_config, render_mode="human")
        policy = OpportunisticPolicy(env)

        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)
        _obs, info = env.reset()

        with simulation_context(env):
            # steps = renderer_config.num_steps
            steps = renderer_config.num_steps
            for _step in range(steps):
                _actions = policy.predict(_obs)
                _obs, rewards, terminals, truncations, info = env.step(_actions)
                _agent_obj = next(
                    (o for o in env.grid_objects.values() if o.get("agent_id") == 0)
                )
                _inv = {
                    env.resource_names[idx]: count
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
    OpportunisticPolicy,
    contextlib,
    display,
    mg_config,
    eval_button,
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
        eval_env = MettaGridEnv(mg_config, render_mode="human")
        eval_policy = OpportunisticPolicy(eval_env)

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
            eval_env.resource_names[idx]: cnt
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
    EvaluatorConfig,
    TrainTool,
    TrainerConfig,
    datetime,
    mg_config,
    env_curriculum,
    logging,
    mo,
    multiprocessing,
    os,
    train_button,
    training_context,
):
    username = os.environ.get("USER", "metta_user")

    def train_agent():
        # Unique run name (so multiple notebook runs don't collide)
        run_name = f"{username}.hello_world_train.mine.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🚀 Starting training run: {run_name}")

        # Create trainer configuration to reach peak performance before unlearning
        trainer_config = TrainerConfig(
            total_timesteps=2200000,  # Train to 2.2M to reach peak performance (~12-13 ore)
            batch_size=32768,  # Reduced batch size for more stable learning
            minibatch_size=256,  # Smaller minibatches for better gradient estimates
            rollout_workers=min(
                4, multiprocessing.cpu_count()
            ),  # Cap workers to prevent resource contention
            forward_pass_minibatch_target_size=256,
            # Use lower learning rate from the start to prevent aggressive updates
            optimizer={
                "learning_rate": 0.0002,  # Lower than default to prevent unlearning
            },
            # More conservative PPO settings
            ppo={
                "clip_coef": 0.15,  # Slightly higher clip to prevent too aggressive updates
                "ent_coef": 0.01,  # Higher entropy to maintain exploration
                "target_kl": 0.015,  # Add KL divergence limit to prevent large policy updates
            },
            checkpoint=CheckpointConfig(
                checkpoint_interval=20,  # Frequent checkpoints to catch peak performance
                remote_prefix=f"s3://softmax-public/policies/{run_name}",
            ),
        )

        training_env_cfg = TrainingEnvironmentConfig(
            curriculum=env_curriculum(mg_config)
        )

        evaluator_cfg = EvaluatorConfig(
            epoch_interval=20,  # Frequent evaluation to monitor for unlearning
            evaluate_remote=False,
            evaluate_local=True,
            replay_dir=f"s3://softmax-public/replays/{run_name}",
        )

        # Create and configure the training tool
        train_tool = TrainTool(
            trainer=trainer_config,
            training_env=training_env_cfg,
            evaluator=evaluator_cfg,
            # wandb=WandbConfigOff(),  # Disable wandb for simplicity
            run=run_name,
            run_dir=f"train_dir/{run_name}",
            disable_macbook_optimize=True,
        )

        # Set up logging to capture output
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        try:
            print("🏋️ Training started...")
            with training_context():
                result = train_tool.invoke(args={}, overrides=[])
                print(f"✅ Training completed successfully! Result: {result}")
            return run_name
        except Exception as e:
            print(f"❌ Training failed: {e}")

            traceback.print_exc()

    mo.stop(not train_button.value)
    run_name = train_agent()
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
    MettaGridEnv,
    Path,
    CheckpointManager,
    WandbConfig,
    contextlib,
    display,
    mg_config,
    eval_trained_button,
    get_repo_root,
    io,
    mo,
    np,
    os,
    pd,
    renderer_config,
    run_name,
    simulation_context,
    time,
    torch,
    widgets,
):
    mo.stop(not eval_trained_button.value)

    def evaluate_agent():
        """
        Fixed simplified version to test path resolution and policy loading
        """
        # Change to repo root directory so relative paths work correctly

        original_cwd = os.getcwd()
        repo_root = get_repo_root()
        os.chdir(repo_root)

        try:
            # Find all checkpoints and select the best one (not just latest)
            ckpt_dir = Path("train_dir") / run_name / "checkpoints"
            print(f"Looking for checkpoints in: {ckpt_dir.absolute()}")
            print(f"Directory exists: {ckpt_dir.exists()}")
            if ckpt_dir.exists():
                checkpoints = list(ckpt_dir.glob("*.pt"))
                print(
                    f"Found {len(checkpoints)} checkpoint files: {[c.name for c in checkpoints]}"
                )
            if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pt")):
                raise Exception(f"No checkpoints found in {ckpt_dir.absolute()}")

            # Get all checkpoints sorted by epoch number (extract epoch from filename)
            checkpoints = list(ckpt_dir.glob("*.pt"))
            if (
                len(checkpoints) > 10
            ):  # If we have many checkpoints, try one from the peak learning phase
                # Sort by epoch number and take one from around 60-80% through training
                checkpoints.sort(
                    key=lambda p: int("".join(filter(str.isdigit, p.stem)))
                    if any(c.isdigit() for c in p.stem)
                    else 0
                )
                peak_idx = int(
                    len(checkpoints) * 0.7
                )  # Use checkpoint from 70% through training
                latest_ckpt = checkpoints[peak_idx]
                print(
                    f"Using peak performance checkpoint: {latest_ckpt.name} (index {peak_idx}/{len(checkpoints)})"
                )
                print(
                    f"   📊 This avoids the unlearning phase seen in later checkpoints"
                )
            else:
                latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Using latest checkpoint: {latest_ckpt.name}")

            print(f"Evaluating checkpoint: {latest_ckpt.name}")

            checkpoint_uri = CheckpointManager.normalize_uri(str(latest_ckpt))

            metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)
            run_name_from_ckpt = metadata["run_name"]

            trained_policy = CheckpointManager.load_from_uri(checkpoint_uri)

            # Create evaluation environment
            with contextlib.redirect_stdout(io.StringIO()):
                eval_env = MettaGridEnv(mg_config, render_mode="human")

            # Set device to CPU for evaluation
            trained_policy = trained_policy.to(torch.device("cpu"))

            # Run animated evaluation with the trained policy
            trained_scores: list[int] = []
            trained_ore_scores: list[int] = []

            # Create header and display widgets for animation
            header = widgets.HTML()
            map_box = widgets.HTML()
            display(header, map_box)

            EVAL_EPISODES = 10
            print(f"🎯 Running {EVAL_EPISODES} episodes with animated evaluation...")

            with simulation_context(eval_env):
                for ep in range(1, EVAL_EPISODES + 1):
                    header.value = f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Evaluating trained agent..."

                    _obs, _ = eval_env.reset()

                    steps = (
                        mg_config.game.max_steps
                    )  # Use same steps as training (5000)
                    print(
                        f"Episode {ep}: Running evaluation for {steps} steps (matching training configuration)"
                    )
                    for _step in range(steps):
                        # Use proper observation processing pipeline that matches training
                        td = obs_to_td(_obs, torch.device("cpu"))

                        # The dimension fix in simulation.py ensures proper tensor shapes automatically

                        trained_policy(td)
                        _actions = td["actions"].cpu().numpy()

                        _obs, _, _, _, _ = eval_env.step(_actions)

                        # Update display every few steps to show animation
                        _agent_obj = next(
                            (
                                o
                                for o in eval_env.grid_objects.values()
                                if o.get("agent_id") == 0
                            )
                        )
                        _inv = {
                            eval_env.resource_names[idx]: cnt
                            for idx, cnt in _agent_obj.get("inventory", {}).items()
                        }
                        ore_count = _inv.get("ore_red", 0)
                        battery_count = _inv.get("battery_red", 0)
                        # Calculate reward using training config: ore_red=0.1, battery_red=0.8
                        total_reward = ore_count * 0.1 + battery_count * 0.8
                        header.value = (
                            f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Step {_step + 1}/{steps} - "
                            f"<br />"
                            f"<b>Ore:</b> {ore_count} <b>Reward:</b> {total_reward:.1f}"
                        )
                        with contextlib.redirect_stdout(io.StringIO()) as buffer:
                            buffer_str = eval_env.render()
                        map_box.value = f"<pre>{buffer_str}</pre>"
                        if _step % 500 == 0:  # Print progress every 500 steps
                            print(
                                f"  Episode {ep}, Step {_step}: Ore={ore_count}, Total Reward={total_reward:.1f}"
                            )
                        time.sleep(
                            renderer_config.sleep_time / 50
                        )  # Small delay for animation

                    # Final inventory count for this episode - use total reward, not just ore
                    _agent_obj = next(
                        (
                            o
                            for o in eval_env.grid_objects.values()
                            if o.get("agent_id") == 0
                        )
                    )
                    _inv = {
                        eval_env.resource_names[idx]: cnt
                        for idx, cnt in _agent_obj.get("inventory", {}).items()
                    }
                    ore_count = int(_inv.get("ore_red", 0))
                    battery_count = int(_inv.get("battery_red", 0))
                    # Calculate total reward based on training reward structure
                    total_reward = ore_count * 0.1 + battery_count * 0.8
                    trained_scores.append(total_reward)
                    trained_ore_scores.append(ore_count)

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
    Best Episode: {max(trained_ore_scores)} ore
    Worst Episode: {min(trained_scores)} ore

    Individual Episode Scores: {trained_scores}

    Compare this to the opportunistic baseline from earlier!
    </pre>"""

            display(
                pd.DataFrame(
                    {
                        "episode": list(range(1, EVAL_EPISODES + 1)),
                        "total_reward": trained_scores,
                        "running_avg": running_avg,
                        "ore": trained_ore_scores,
                        "running_avg_ore": pd.Series(trained_ore_scores)
                        .expanding()
                        .mean(),
                    },
                )
            )

            print(
                f"\n🎯 Trained agent performance: {mean_score:.2f} ± {std_score:.2f} total reward"
            )
            print(
                f"    (Reward = ore_count * 0.1 + battery_count * 0.8, matching training config)"
            )
            print(f"📊 Compare with opportunistic baseline from earlier evaluation!")
            print(f"\n📋 TRAINING vs EVALUATION COMPARISON:")
            print(f"   - Training WandB shows: ~40+ ore per episode")
            print(
                f"   - Evaluation shows: {mean_score:.2f} total reward ({mean_score / 0.1:.1f} ore equivalent)"
            )
            print(f"   - Episode length: {steps} steps (matching training max_steps)")
            if mean_score < 2.0:  # Less than 20 ore equivalent
                print(
                    f"   ⚠️  MISMATCH: Evaluation performance much lower than training metrics!"
                )
                print(
                    f"   🔍 Possible issues: checkpoint selection, environment differences, or step count"
                )
            else:
                print(f"   ✅ Performance matches training expectations!")

        finally:
            os.chdir(original_cwd)

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


@app.cell
def _(mo, run_name, show_replay):
    mo.stop(not run_name)

    try:
        # Show the latest replay from the training run
        show_replay(run_name, step="last", width=1250, height=500, autoplay=True)
    except Exception as e:
        print(f"❌ Error loading replay: {e}")
        print("This could mean:")
        print("- Training hasn't generated replays yet (evaluation incomplete)")
        print("- Run not found in W&B (check run name)")
        print("- Network connectivity issues")
        print(f"\nReplays are stored on S3 at: s3://softmax-public/replays/{run_name}/")
    return


@app.cell
def _(mo, wandb):
    def display_by_wandb_path(path: str, *, height: int) -> None:
        """Display a wandb object (usually in an iframe) given its URI.

        Supports runs, sweeps, projects, reports, and other wandb objects.
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

    return (display_by_wandb_path,)


@app.cell
def _(display_by_wandb_path, mo, run_name):
    mo.stop(not run_name)
    display_by_wandb_path(f"metta-research/metta/{run_name}", height=580)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Let's run a new example with a new map

    This time we have a mine (m) on the right to dispense red ore and a red converter (R) on the left to turn the ore into batteries.

    ```text
    ###########
    #R...@...m#
    ###########
    ```

    ###  🔄 Multi-Step Task Success

    This second environment is more complex - it requires:

    1. Collect ore from the mine (right side)
    2. Transport ore to the generator (left side)
    3. Convert ore to batteries for higher rewards
    """
    )
    return


@app.cell
def _(
    AgentRewards,
    AsciiMapBuilder,
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
    mg_config2 = make_arena(num_agents=1)

    # Replace with our simple hallway map
    map_data2 = [list(line) for line in hallway_map2.splitlines()]
    mg_config2.game.map_builder = AsciiMapBuilder.Config(map_data=map_data2)

    # Simple customizations
    mg_config2.game.max_steps = 5000
    mg_config2.game.obs_width = 11
    mg_config2.game.obs_height = 11

    # Enable basic movement and item collection - disable combat
    mg_config2.game.actions.move.enabled = True
    mg_config2.game.actions.rotate.enabled = True
    mg_config2.game.actions.noop.enabled = False  # Disable no-op to force action
    mg_config2.game.actions.attack.enabled = False
    mg_config2.game.actions.get_items.enabled = True
    mg_config2.game.actions.put_items.enabled = True
    mg_config2.game.actions.change_color.enabled = False
    mg_config2.game.actions.change_glyph.enabled = False
    mg_config2.game.actions.swap.enabled = False

    # CONVERSION INCENTIVE: Make conversion much more profitable than resource limit camping
    mg_config2.game.agent.rewards = AgentRewards(
        inventory={
            "ore_red": -0.02,
            "battery_red": 5.0,
        },
        inventory_max={
            "ore_red": 255,
            "battery_red": 255,
        },
    )

    # Force more frequent conversion by limiting ore storage
    mg_config2.game.agent.resource_limits = {"ore_red": 10}  # Can only hold 10 ore max

    # Use action failure penalty for efficiency (encourages purposeful movement)
    mg_config2.game.agent.action_failure_penalty = 0.01

    renderer_config2 = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=3000,
        sleep_time=0.0020,
        renderer_type="human",
    )
    return mg_config2, renderer_config2


@app.cell
def _(mo):
    observe_button2 = mo.ui.run_button(label="Click to run observation below")
    observe_button2
    return (observe_button2,)


@app.cell
def _(
    MettaGridEnv,
    OpportunisticPolicy,
    contextlib,
    display,
    mg_config2,
    io,
    mo,
    observe_button2,
    renderer_config2,
    simulation_context,
    time,
    widgets,
):
    mo.stop(not observe_button2.value)

    def observe_agent2():
        # Create environment with proper MettaGridConfig
        env = MettaGridEnv(mg_config2, render_mode="human")
        policy = OpportunisticPolicy(env)

        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)
        _obs, info = env.reset()

        with simulation_context(env):
            # steps = renderer_config.num_steps
            steps = mg_config2.game.max_steps
            for _step in range(steps):
                _actions = policy.predict(_obs)
                _obs, rewards, terminals, truncations, info = env.step(_actions)
                _agent_obj = next(
                    (o for o in env.grid_objects.values() if o.get("agent_id") == 0)
                )
                _inv = {
                    env.resource_names[idx]: count
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
    eval_button2 = mo.ui.run_button(label="Click to run another evaluation")
    eval_button2
    return (eval_button2,)


@app.cell
def _(
    EVAL_EPISODES,
    MettaGridEnv,
    OpportunisticPolicy,
    contextlib,
    display,
    mg_config2,
    eval_button2,
    io,
    mo,
    np,
    pd,
    renderer_config2,
):
    mo.stop(not eval_button2.value)

    def _():
        scores_ore: list[int] = []
        scores_batteries: list[int] = []

        with contextlib.redirect_stdout(io.StringIO()):
            # Create evaluation environment with our simple config
            eval_env = MettaGridEnv(mg_config2, render_mode="human")
            eval_policy = OpportunisticPolicy(eval_env)

        for ep in range(1, EVAL_EPISODES + 1):
            _obs, _ = eval_env.reset()
            inv_count = 0
            for _step in range(renderer_config2.num_steps):
                _actions = eval_policy.predict(_obs)
                _obs, _, _, _, _ = eval_env.step(_actions)
            _agent_obj = next(
                (o for o in eval_env.grid_objects.values() if o.get("agent_id") == 0)
            )
            _inv = {
                eval_env.resource_names[idx]: cnt
                for idx, cnt in _agent_obj.get("inventory", {}).items()
            }
            inv_count_ore = int(_inv.get("ore_red", 0))
            inv_count_batteries = int(_inv.get("battery_red", 0))
            scores_ore.append(inv_count_ore)
            scores_batteries.append(inv_count_batteries)

        mean_score_ore = np.mean(scores_ore)
        mean_score_batteries = np.mean(scores_batteries)
        std_score_ore = np.std(scores_ore)
        std_score_batteries = np.std(scores_batteries)
        running_avg_ore = pd.Series(scores_ore).expanding().mean()
        running_avg_batteries = pd.Series(scores_batteries).expanding().mean()

        analysis_str = (
            f"Opportunistic agent baseline: {mean_score_ore:.2f} ± {std_score_ore:.2f} ore collected, "
            f"{mean_score_batteries:.2f} ± {std_score_batteries:.2f} batteries collected"
        )
        print(analysis_str)
        display(mo.md(analysis_str))
        display(
            pd.DataFrame(
                {
                    "episode": list(range(1, EVAL_EPISODES + 1)),
                    "ore_red": scores_ore,
                    "battery_red": scores_batteries,
                    "running_avg_ore": running_avg_ore,
                    "running_avg_battery": running_avg_batteries,
                }
            )
        )
        eval_env.close()

    _()
    return


@app.cell
def _(mo):
    train_button2 = mo.ui.run_button(label="Click to run training below")
    train_button2
    return (train_button2,)


@app.cell
def _(
    CheckpointConfig,
    EvaluatorConfig,
    TrainingEnvironmentConfig,
    TrainTool,
    TrainerConfig,
    datetime,
    mg_config2,
    env_curriculum,
    logging,
    mo,
    multiprocessing,
    train_button2,
    training_context,
    username,
):
    mo.stop(not train_button2.value)

    def train_agent2():
        # Create a simple curriculum with our hallway environment
        curriculum = env_curriculum(mg_config2)

        run_name2 = f"{username}.hello_world_train.mine_plus_generator.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trainer_config = TrainerConfig(
            total_timesteps=3500000,  # Extended training to master conversion cycles
            batch_size=65536,  # Larger batches for stable learning of clear signal
            minibatch_size=512,  # Bigger minibatches with clean reward structure
            rollout_workers=min(
                6, multiprocessing.cpu_count()
            ),  # More workers for route exploration
            forward_pass_minibatch_target_size=512,
            # Learning rate with decay to prevent unlearning
            optimizer={
                "learning_rate": 0.0004,  # Start high to learn conversion quickly
                "eps": 1e-8,  # Numerical stability
                "weight_decay": 5e-7,  # Slightly higher regularization to prevent overfitting
            },
            # Anti-unlearning PPO settings
            ppo={
                "clip_coef": 0.15,  # Tighter clipping to prevent policy drift
                "ent_coef": 0.02,  # Start with exploration, will decay via scheduler
                "target_kl": 0.03,  # Stricter KL limit to prevent large policy changes
                "vf_coef": 0.5,  # Standard value function coefficient
                "gamma": 0.99,  # Standard discount for immediate conversion feedback
                "gae_lambda": 0.95,  # Standard GAE
                "max_grad_norm": 0.5,  # Gradient clipping for stability
            },
            # Comprehensive scheduling to prevent unlearning
            hyperparameter_scheduler={
                "learning_rate_schedule": {
                    "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
                    "initial_value": 0.0004,  # Start high for initial learning
                    "min_value": 0.00005,  # Decay to very low to lock in behavior
                },
                "ppo_ent_coef_schedule": {
                    "_target_": "metta.rl.hyperparameter_scheduler.LinearSchedule",
                    "initial_value": 0.02,  # Start with exploration
                    "min_value": 0.001,  # Reduce exploration once learned
                },
            },
            checkpoint=CheckpointConfig(
                checkpoint_interval=10,  # More frequent checkpoints to catch peak
                remote_prefix=f"s3://softmax-public/policies/{run_name2}",
            ),
        )

        training_env_cfg = TrainingEnvironmentConfig(curriculum=curriculum)

        evaluator_cfg = EvaluatorConfig(
            epoch_interval=10,  # More frequent evaluation to monitor unlearning
            evaluate_remote=False,
            evaluate_local=True,
            replay_dir=f"s3://softmax-public/replays/{run_name2}",
        )

        # Create and configure the training tool
        train_tool = TrainTool(
            trainer=trainer_config,
            training_env=training_env_cfg,
            evaluator=evaluator_cfg,
            # wandb=WandbConfigOff(),  # Disable wandb for simplicity
            run=run_name2,
            run_dir=f"train_dir/{run_name2}",
            disable_macbook_optimize=True,
        )

        # Set up logging to capture output
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        try:
            print("🏋️ Training started...")
            with training_context():
                result = train_tool.invoke(args={}, overrides=[])
                print(f"✅ Training completed successfully! Result: {result}")
        except Exception as e:
            print(f"❌ Training failed: {e}")

            traceback.print_exc()

        return run_name2

    run_name2 = train_agent2()
    return (run_name2,)


@app.cell
def _(mo):
    eval_trained_button2 = mo.ui.run_button(label="Click to evaluate 2nd trained agent")
    eval_trained_button2
    return (eval_trained_button2,)


@app.cell
def _(
    EVAL_EPISODES,
    MettaGridEnv,
    Path,
    CheckpointManager,
    TensorDict,
    WandbConfig,
    contextlib,
    display,
    mg_config,
    mg_config2,
    eval_trained_button2,
    io,
    mo,
    np,
    pd,
    renderer_config2,
    run_name2,
    simulation_context,
    time,
    torch,
    widgets,
):
    mo.stop(not eval_trained_button2.value or not run_name2)

    # run_name2 = "zfogg.hello_world_train.mine_plus_generator.20250827_042602"

    def evaluate_agent2():
        # Find all checkpoints and select the best one using peak performance strategy
        ckpt_dir = Path("train_dir") / run_name2 / "checkpoints"
        print(f"Looking for checkpoints in: {ckpt_dir.absolute()}")
        print(f"Directory exists: {ckpt_dir.exists()}")
        if ckpt_dir.exists():
            checkpoints = list(ckpt_dir.glob("*.pt"))
            print(
                f"Found {len(checkpoints)} checkpoint files: {[c.name for c in checkpoints]}"
            )
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pt")):
            raise Exception(f"No checkpoints found in {ckpt_dir.absolute()}")

        # Get all checkpoints sorted by epoch number (extract epoch from filename)
        checkpoints = list(ckpt_dir.glob("*.pt"))
        if (
            len(checkpoints) > 10
        ):  # If we have many checkpoints, try one from the peak learning phase
            # Sort by epoch number and take one from around 60-80% through training
            checkpoints.sort(
                key=lambda p: int("".join(filter(str.isdigit, p.stem)))
                if any(c.isdigit() for c in p.stem)
                else 0
            )
            peak_idx = int(
                len(checkpoints) * 0.7
            )  # Use checkpoint from 70% through training
            latest_ckpt = checkpoints[peak_idx]
            print(
                f"Using peak performance checkpoint: {latest_ckpt.name} (index {peak_idx}/{len(checkpoints)})"
            )
            print(f"   📊 This avoids the unlearning phase seen in later checkpoints")
        else:
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Using latest checkpoint: {latest_ckpt.name}")

        print(f"Evaluating checkpoint: {latest_ckpt.name}")

        checkpoint_uri = CheckpointManager.normalize_uri(str(latest_ckpt))

        metadata = CheckpointManager.get_policy_metadata(checkpoint_uri)
        run_name_from_ckpt = metadata["run_name"]

        trained_policy = CheckpointManager.load_from_uri(checkpoint_uri)

        # Create evaluation environment
        with contextlib.redirect_stdout(io.StringIO()):
            eval_env = MettaGridEnv(mg_config2, render_mode="human")

        # Set device to CPU for evaluation
        trained_policy = trained_policy.to(torch.device("cpu"))

        # Run animated evaluation with the trained policy
        trained_scores: list[int] = []
        trained_scores_ore: list[int] = []
        trained_scores_batteries: list[int] = []

        # Create header and display widgets for animation
        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)

        print(f"🔧 EVALUATION SETUP:")
        print(f"   - Environment max_steps: {mg_config.game.max_steps}")
        print(f"   - Selected checkpoint: {latest_ckpt.name}")

        with simulation_context(eval_env):
            print(f"🎯 Running {EVAL_EPISODES} episodes with animated evaluation...")
            for ep in range(1, EVAL_EPISODES + 1):
                header.value = (
                    f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Evaluating trained agent..."
                )

                _obs, _ = eval_env.reset()
                # Convert obs to tensor format for policy
                obs_tensor = torch.as_tensor(_obs, device=torch.device("cpu"))

                steps = mg_config2.game.max_steps  # Use same steps as training (5000)
                for _step in range(steps):  # Same number of steps as opportunistic
                    # Use TensorDict format for trained policy (same as simulation.py:272-275)
                    td = TensorDict(
                        {"env_obs": obs_tensor}, batch_size=obs_tensor.shape[0]
                    )
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
                        eval_env.resource_names[idx]: cnt
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
                    time.sleep(
                        renderer_config2.sleep_time / 100
                    )  # Small delay for animation

                # Final inventory count for this episode
                _agent_obj = next(
                    (
                        o
                        for o in eval_env.grid_objects.values()
                        if o.get("agent_id") == 0
                    )
                )
                _inv = {
                    eval_env.resource_names[idx]: cnt
                    for idx, cnt in _agent_obj.get("inventory", {}).items()
                }
                inv_count_ore = int(_inv.get("ore_red", 0))
                inv_count_batteries = int(_inv.get("battery_red", 0))
                reward_ore = (
                    inv_count_ore * mg_config2.game.agent.rewards.inventory.ore_red
                )  # Will be 0.0
                reward_batteries = (
                    inv_count_batteries
                    * mg_config2.game.agent.rewards.inventory.battery_red
                )  # Only source of reward
                reward = reward_ore + reward_batteries  # Pure battery reward
                trained_scores.append(reward)
                trained_scores_ore.append(inv_count_ore)
                trained_scores_batteries.append(inv_count_batteries)

            eval_env.close()

        # Calculate and display results
        mean_score = np.mean(trained_scores)
        mean_score_ore = np.mean(trained_scores_ore)
        mean_score_batteries = np.mean(trained_scores_batteries)
        std_score = np.std(trained_scores)
        std_score_ore = np.std(trained_scores_ore)
        std_score_batteries = np.std(trained_scores_batteries)
        running_avg = pd.Series(trained_scores).expanding().mean()
        running_avg_ore = pd.Series(trained_scores_ore).expanding().mean()
        running_avg_batteries = pd.Series(trained_scores_batteries).expanding().mean()

        # Show final results
        header.value = f"<b>✅ Evaluation Complete!</b>"
        map_box.value = f"""<pre>
    🏆 TRAINED AGENT RESULTS 🏆

    Episodes: {EVAL_EPISODES}

    📊 COMPREHENSIVE STATISTICS:
    Total Reward:  {mean_score:.2f} ± {std_score:.2f}
    Ore Collected: {mean_score_ore:.2f} ± {std_score_ore:.2f}
    Batteries:     {mean_score_batteries:.2f} ± {std_score_batteries:.2f}

    🎯 BEST EPISODE:
    Reward: {max(trained_scores):.1f}, Ore: {max(trained_scores_ore)}, Batteries: {max(trained_scores_batteries)}

    ⚠️ WORST EPISODE:
    Reward: {min(trained_scores):.1f}, Ore: {min(trained_scores_ore)}, Batteries: {min(trained_scores_batteries)}

    💡 Multi-step task: Collect ore → Transport → Convert to batteries
    Compare this to the opportunistic baseline from earlier!
        </pre>"""

        display(
            pd.DataFrame(
                {
                    "episode": list(range(1, EVAL_EPISODES + 1)),
                    "reward": trained_scores,
                    "ore_red": trained_scores_ore,
                    "battery_red": trained_scores_batteries,
                    "running_avg_reward": running_avg,
                    "running_avg_ore": running_avg_ore,
                    "running_avg_batteries": running_avg_batteries,
                }
            )
        )

        print(f"\n🎯 Trained agent performance:")
        print(f"   Total Reward: {mean_score:.2f} ± {std_score:.2f}")
        print(f"   Ore Collected: {mean_score_ore:.2f} ± {std_score_ore:.2f}")
        print(
            f"   Batteries Collected: {mean_score_batteries:.2f} ± {std_score_batteries:.2f}"
        )
        print(f"\n💰 Reward Breakdown:")
        print(
            f"   Ore reward rate: {mg_config2.game.agent.rewards.inventory.ore_red} per ore"
        )
        print(
            f"   Battery reward rate: {mg_config2.game.agent.rewards.inventory.battery_red} per battery"
        )
        print(
            f"   Average ore reward: {mean_score_ore * mg_config2.game.agent.rewards.inventory.ore_red:.2f}"
        )
        print(
            f"   Average battery reward: {mean_score_batteries * mg_config2.game.agent.rewards.inventory.battery_red:.2f}"
        )
        print(f"📊 Compare with opportunistic baseline from earlier evaluation!")

    evaluate_agent2()
    return


@app.cell
def _(mo, run_name2, show_replay):
    mo.stop(not run_name2)

    try:
        # Show the latest replay from the training run
        show_replay(run_name2, step="last", width=1250, height=500, autoplay=True)
    except Exception as e:
        print(f"❌ Error loading replay: {e}")
        print("This could mean:")
        print("- Training hasn't generated replays yet (evaluation incomplete)")
        print("- Run not found in W&B (check run name)")
        print("- Network connectivity issues")
        print(
            f"\nReplays are stored on S3 at: s3://softmax-public/replays/{run_name2}/"
        )
    return


@app.cell
def _(display_by_wandb_path, mo, run_name2):
    mo.stop(not run_name2)
    display_by_wandb_path(f"metta-research/metta/{run_name2}", height=580)
    return


if __name__ == "__main__":
    app.run()
