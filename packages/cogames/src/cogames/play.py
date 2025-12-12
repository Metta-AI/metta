"""Game playing functionality for CoGames."""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout

logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_spec: PolicySpec,
    game_name: str,
    seed: int = 42,
    render_mode: RenderMode = "gui",
    save_replay: Optional[Path] = None,
) -> None:
    """Play a single game episode with a policy.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_spec: Policy specification (class path and optional data path)
        game_name: Human-readable name of the game (used for logging/metadata)
        seed: Random seed
        render_mode: Render mode - "gui", "unicode", or "none"
        save_replay: Optional directory path to save replay. Directory will be created if it doesn't exist.
            Replay will be saved with a unique UUID-based filename.
    """

    logger.debug("Starting play session", extra={"game_name": game_name})

    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    policy = initialize_or_load_policy(policy_env_info, policy_spec)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)]

    # Set up replay writer if requested
    event_handlers = []
    replay_writer = None
    if save_replay:
        replay_writer = ReplayLogWriter(str(save_replay))
        event_handlers.append(replay_writer)

    # Create simulator and renderer
    rollout = Rollout(
        env_cfg,
        agent_policies,
        render_mode=render_mode,
        seed=seed,
        event_handlers=event_handlers,
    )
    rollout.run_until_done()

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {rollout._sim.current_step}")
    console.print(f"Total Rewards: {rollout._sim.episode_rewards}")
    console.print(f"Final Reward Sum: {float(sum(rollout._sim.episode_rewards)):.2f}")

    # Print replay command if replay was saved
    if replay_writer:
        for replay_path in replay_writer.get_written_replay_paths():
            console.print("\n[bold cyan]Replay saved![/bold cyan]")
            console.print("To watch the replay, run:")
            console.print(f"[bold green]cogames replay {replay_path}[/bold green]")
