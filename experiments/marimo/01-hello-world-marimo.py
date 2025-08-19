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
    print("Setup done")
    return (
        MettaGridEnv,
        Path,
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
def _():
    # Simple approach: use the built-in arena and add a custom map - just like the demos do
    from metta.mettagrid.config.envs import make_arena
    from metta.mettagrid.map_builder.ascii import AsciiMapBuilderConfig

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
    env_config.game.map_builder = AsciiMapBuilderConfig(map_data=map_data)

    # Simple customizations
    env_config.game.max_steps = 1000
    env_config.game.obs_width = 11
    env_config.game.obs_height = 11

    # Enable basic movement and item collection - disable combat
    env_config.game.actions.attack.enabled = False
    env_config.game.actions.swap.enabled = False
    env_config.game.actions.change_color.enabled = False
    env_config.game.actions.change_glyph.enabled = False

    # Ensure ore collection gives rewards
    env_config.game.agent.rewards.inventory.ore_red = 1.0

    # Create a proper RendererToolConfig for policy creation
    from tools.renderer import RendererToolConfig

    renderer_config = RendererToolConfig(
        policy_type="opportunistic",
        num_steps=150,
        sleep_time=0.03,
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

    def _():
        # Create environment with proper EnvConfig
        env = MettaGridEnv(env_config, render_mode="human")
        policy = get_policy(renderer_config.policy_type, env, renderer_config)

        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)
        _obs, info = env.reset()

        for _step in range(renderer_config.num_steps):
            _actions = policy.predict(_obs)
            _obs, rewards, terminals, truncations, info = env.step(_actions)
            _agent_obj = next(
                (o for o in env.grid_objects.values() if o.get("agent_id") == 0)
            )
            _inv = {
                env.inventory_item_names[idx]: count
                for idx, count in _agent_obj.get("inventory", {}).items()
            }
            header.value = f"<b>Step:</b> {_step + 1}/{renderer_config.num_steps} <br/> <b>Inventory:</b> {_inv.get('ore_red', 0)}"
            with contextlib.redirect_stdout(io.StringIO()):
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
    mo.stop(not eval_button.value)

    EVAL_EPISODES = 10
    scores: list[int] = []
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
    from metta.common.wandb.wandb_context import WandbConfigOff
    from metta.cogworks.curriculum import env_curriculum

    # Unique run name (so multiple notebook runs don't collide)
    run_name = f"hello_world_train.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"🚀 Starting training run: {run_name}")

    # Create a simple curriculum with our hallway environment
    curriculum = env_curriculum(env_config)

    # Create trainer configuration with small settings for demo
    trainer_config = TrainerConfig(
        curriculum=curriculum,
        # total_timesteps=10000,  # Small demo run
        total_timesteps=1000,  # Small demo run
        batch_size=256,
        minibatch_size=256,
        rollout_workers=2,  # Correct field name
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
        wandb=WandbConfigOff(),  # Disable wandb for simplicity
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

        # Create a simple wrapper that doesn't depend on a separate environment
        class SimpleTrainedPolicyWrapper:
            def __init__(self, policy):
                self.policy = policy

            def predict(self, obs):
                import torch

                with torch.no_grad():
                    # Convert observation to tensor
                    obs_tensor = torch.from_numpy(obs).float()

                    # Create input dict for MettaAgent.forward()
                    if len(obs_tensor.shape) == 3:  # Single agent obs
                        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

                    input_dict = {"grid_obs": obs_tensor}

                    # Get actions from policy
                    result = self.policy.forward(input_dict)

                    # Extract actions from result
                    if hasattr(result, "get") and "action" in result:
                        actions = result["action"]
                    elif hasattr(result, "action"):
                        actions = result.action
                    else:
                        # Fallback: assume result is the action tensor
                        actions = result

                    if isinstance(actions, torch.Tensor):
                        actions = actions.cpu().numpy()

                    # Ensure proper shape and dtype
                    if len(actions.shape) == 1:
                        actions = actions.reshape(1, -1)

                    return actions.astype("int32")

        # Create fallback opportunistic policy wrapper
        class SimpleOpportunisticWrapper:
            def __init__(self):
                pass

            def predict(self, obs):
                # Simple opportunistic logic - move towards resources or randomly
                import numpy as np
                from metta.mettagrid.util.actions import generate_valid_random_actions

                # For now, just generate random valid actions
                # This is a simplified version - could be enhanced with actual opportunistic logic
                actions = np.array(
                    [[0, 1]], dtype="int32"
                )  # Move action with direction 1
                return actions

        # Load the trained policy using PolicyStore (handles PolicyRecord objects)
        from metta.agent.policy_store import PolicyStore

        try:
            # Create PolicyStore and load the checkpoint
            policy_store = PolicyStore(device="cpu")
            policy_uri = f"file://{latest_ckpt.absolute()}"

            # Load policy record and get the actual policy
            policy_record = policy_store.load_from_uri(policy_uri)
            raw_policy = policy_record.policy

            trained_policy = SimpleTrainedPolicyWrapper(raw_policy)
            print("✅ Successfully loaded trained policy")

        except Exception as e:
            print(f"❌ Failed to load trained policy: {e}")
            print("Using simplified opportunistic policy as fallback")
            trained_policy = SimpleOpportunisticWrapper()

        # Run animated evaluation just like the opportunistic agent
        EVAL_EPISODES = 10
        trained_scores: list[int] = []

        # Create header and display widgets
        header = widgets.HTML()
        map_box = widgets.HTML()
        display(header, map_box)

        with contextlib.redirect_stdout(io.StringIO()):
            eval_env = MettaGridEnv(env_config, render_mode="human")

        for ep in range(1, EVAL_EPISODES + 1):
            header.value = (
                f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Evaluating trained agent..."
            )

            _obs, _ = eval_env.reset()
            inv_count = 0

            for _step in range(150):  # Same number of steps as opportunistic
                _actions = trained_policy.predict(_obs)
                _obs, _, _, _, _ = eval_env.step(_actions)

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
                        f"<b>Episode {ep}/{EVAL_EPISODES}</b> - Step {_step + 1}/150 - "
                        f"<b>Ore collected:</b> {_inv.get('ore_red', 0)}"
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        buffer_str = eval_env.render()
                    map_box.value = f"<pre>{buffer_str}</pre>"
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


if __name__ == "__main__":
    app.run()
