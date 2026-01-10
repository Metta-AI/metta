"""Game playing functionality for CoGames."""

import logging
import uuid
from pathlib import Path
from typing import Optional, cast

from alo.pure_single_episode_runner import PureSingleEpisodeSpecJob, run_pure_single_episode_from_specs
from rich.console import Console

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.replay_log_writer import EpisodeReplay

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

    replay_path = None
    if save_replay:
        save_replay.mkdir(parents=True, exist_ok=True)
        replay_path = save_replay / f"{uuid.uuid4()}.json.z"

    job = PureSingleEpisodeSpecJob(
        policy_specs=[policy_spec],
        assignments=[0] * env_cfg.game.num_agents,
        env=env_cfg,
        replay_uri=str(replay_path) if replay_path else None,
        seed=seed,
    )
    try:
        results, replay = run_pure_single_episode_from_specs(job, device="cpu", render_mode=render_mode)
    except KeyboardInterrupt:
        logger.info("Interrupted; ending episode early.")
        return

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {results.steps}")
    console.print(f"Total Rewards: {results.rewards}")
    console.print(f"Final Reward Sum: {float(sum(results.rewards)):.2f}")

    # Print replay command if replay was saved
    if replay_path:
        replay = cast(EpisodeReplay, replay)
        if str(replay_path).endswith(".gz"):
            replay.set_compression("gzip")
        elif str(replay_path).endswith(".z"):
            replay.set_compression("zlib")
        replay.write_replay(str(replay_path))

        console.print("\n[bold cyan]Replay saved![/bold cyan]")
        console.print("To watch the replay, run:")
        console.print(f"[bold green]cogames replay {replay_path}[/bold green]")
