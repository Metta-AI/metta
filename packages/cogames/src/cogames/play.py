"""Game playing functionality for CoGames."""

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cogames.game import get_game
from cogames.policy import Policy, create_policy
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.mettagrid_env import MettaGridEnv


class GameRunner:
    """Handles running games with different policies."""

    def __init__(
        self,
        config: MettaGridConfig,
        console: Optional[Console] = None,
        render: bool = False,
        save_video: Optional[Path] = None,
    ):
        """Initialize the game runner.

        Args:
            config: Game configuration
            console: Optional Rich console for output
            render: Whether to render the game
            save_video: Optional path to save video
        """
        self.config = config
        self.console = console or Console()
        self.render = render
        self.save_video = save_video
        self.env = None

    def setup(self) -> MettaGridEnv:
        """Set up the game environment.

        Returns:
            The initialized environment
        """
        self.env = MettaGridEnv(env_cfg=self.config)
        return self.env

    def run_episode(self, policy: Policy, max_steps: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
        """Run a single episode with the given policy.

        Args:
            policy: The policy to use
            max_steps: Maximum number of steps (None for no limit)
            verbose: Whether to print progress

        Returns:
            Episode statistics
        """
        if self.env is None:
            self.setup()

        obs = self.env.reset()
        policy.reset()

        episode_rewards = []
        step_count = 0
        done = False

        while not done and (max_steps is None or step_count < max_steps):
            # Get action from policy
            actions = policy.get_action(obs)

            # Step the environment
            obs, rewards, dones, truncated, info = self.env.step(actions)

            # Aggregate rewards
            if isinstance(rewards, dict):
                total_reward = sum(rewards.values())
            else:
                total_reward = rewards

            episode_rewards.append(total_reward)

            # Check if done
            if isinstance(dones, dict):
                done = any(dones.values())
            else:
                done = dones

            step_count += 1

            if verbose and step_count % 10 == 0:
                self.console.print(f"Step {step_count}: Reward = {total_reward:.2f}")

            if self.render:
                # Render logic would go here
                pass

        total_reward = sum(episode_rewards)
        avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0

        return {
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "steps": step_count,
            "rewards": episode_rewards,
        }

    def run_interactive(
        self,
        max_steps: int = 1000,
    ) -> None:
        """Run the game in interactive mode where user controls actions.

        Args:
            max_steps: Maximum number of steps
        """
        if self.env is None:
            self.setup()

        self.console.print("[cyan]Starting interactive game session[/cyan]")
        self.console.print("Controls: Use arrow keys to move, 'q' to quit")
        self.console.print("[yellow]Interactive mode not fully implemented yet.[/yellow]")

        # This would implement interactive controls
        # For now, it's a placeholder
        self.console.print("Would run interactive game loop here...")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
            self.env = None


def play_game(
    game_name: str,
    scenario_name: str = None,  # Keep for backward compatibility
    policy: str = "random",
    steps: int = 100,
    episodes: int = 1,
    interactive: bool = False,
    render: bool = False,
    save_video: Optional[Path] = None,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Play a game with the specified configuration.

    Args:
        game_name: Name of the game
        scenario_name: Deprecated, kept for backward compatibility
        policy: Policy to use ("random" or path to checkpoint)
        steps: Maximum steps per episode
        episodes: Number of episodes to run
        interactive: Whether to run in interactive mode
        render: Whether to render the game
        save_video: Optional path to save video
        console: Optional Rich console for output

    Returns:
        Game statistics
    """
    if console is None:
        console = Console()

    # Get game configuration
    game_config = get_game(game_name)

    # Create game runner
    runner = GameRunner(config=game_config, console=console, render=render, save_video=save_video)

    try:
        # Set up environment
        env = runner.setup()

        if interactive:
            runner.run_interactive(max_steps=steps)
            return {"mode": "interactive", "status": "completed"}

        # Create policy
        policy_obj = create_policy(policy, env)

        # Run episodes
        all_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {episodes} episodes...", total=episodes)

            for episode in range(episodes):
                result = runner.run_episode(policy=policy_obj, max_steps=steps, verbose=(episodes == 1))
                all_results.append(result)

                if episodes > 1:
                    console.print(
                        f"Episode {episode + 1}: Total Reward = {result['total_reward']:.2f}, Steps = {result['steps']}"
                    )

                progress.update(task, advance=1)

        # Calculate statistics
        total_rewards = [r["total_reward"] for r in all_results]
        avg_reward = sum(total_rewards) / len(total_rewards)
        min_reward = min(total_rewards)
        max_reward = max(total_rewards)

        stats = {
            "episodes": episodes,
            "average_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "total_steps": sum(r["steps"] for r in all_results),
            "results": all_results,
        }

        # Print summary
        console.print("\n[bold green]Game Complete![/bold green]")
        console.print(f"Average Reward: {avg_reward:.2f}")
        console.print(f"Min Reward: {min_reward:.2f}")
        console.print(f"Max Reward: {max_reward:.2f}")

        return stats

    finally:
        runner.cleanup()


def evaluate_policy(
    game_name: str,
    scenario_name: str = None,  # Keep for backward compatibility
    policy: str = "random",
    episodes: int = 10,
    render: bool = False,
    save_video: Optional[Path] = None,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Evaluate a policy on a game.

    Args:
        game_name: Name of the game
        scenario_name: Deprecated, kept for backward compatibility
        policy: Policy to evaluate (path to checkpoint or "random")
        episodes: Number of evaluation episodes
        render: Whether to render evaluation
        save_video: Optional path to save video
        console: Optional Rich console for output

    Returns:
        Evaluation statistics
    """
    return play_game(
        game_name=game_name,
        scenario_name=scenario_name,
        policy=policy,
        steps=None,  # No step limit for evaluation
        episodes=episodes,
        interactive=False,
        render=render,
        save_video=save_video,
        console=console,
    )
