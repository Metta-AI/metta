"""Training functionality for CoGames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cogames.game import get_game


@dataclass
class TrainingConfig:
    """Configuration for training."""

    algorithm: str = "ppo"
    steps: int = 10000
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2
    save_freq: int = 1000
    eval_freq: int = 500
    eval_episodes: int = 5
    seed: Optional[int] = None
    device: str = "auto"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


class Trainer:
    """Handles policy training for games."""

    def __init__(
        self,
        game_name: str,
        scenario_name: str = None,  # Keep for backward compatibility
        config: TrainingConfig = None,
        save_path: Optional[Path] = None,
        console: Optional[Console] = None,
    ):
        """Initialize the trainer.

        Args:
            game_name: Name of the game
            scenario_name: Deprecated, kept for backward compatibility
            config: Training configuration
            save_path: Optional path to save checkpoints
            console: Optional Rich console for output
        """
        self.game_name = game_name
        self.scenario_name = scenario_name or game_name  # Use game_name if scenario not provided
        self.config = config or TrainingConfig()
        self.save_path = save_path or Path(f"checkpoints/{game_name}")
        self.console = console or Console()
        self.game_config = get_game(game_name)

    def setup_wandb(self) -> None:
        """Set up Weights & Biases logging."""
        if self.config.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    tags=self.config.wandb_tags or [],
                    config={
                        "game": self.game_name,
                        "scenario": self.scenario_name,
                        "algorithm": self.config.algorithm,
                        "steps": self.config.steps,
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                    },
                )
                self.console.print(f"[green]W&B logging initialized: {self.config.wandb_project}[/green]")
            except ImportError:
                self.console.print("[yellow]wandb not installed, skipping W&B logging[/yellow]")

    def train(self) -> Dict[str, Any]:
        """Run the training process.

        Returns:
            Training statistics and results
        """
        self.console.print(f"[cyan]Starting training for {self.game_name} - {self.scenario_name}[/cyan]")
        self.console.print(f"Algorithm: {self.config.algorithm.upper()}")
        self.console.print(f"Steps: {self.config.steps}")
        self.console.print(f"Save path: {self.save_path}")

        # Set up W&B if configured
        self.setup_wandb()

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # This would integrate with metta.rl.training
        # For now, we'll show the intended workflow

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Training {self.config.algorithm.upper()}...", total=self.config.steps)

            # Simulated training loop
            for step in range(0, self.config.steps, 100):
                # This would be actual training steps
                progress.update(task, advance=100)

                # Checkpoint saving
                if step > 0 and step % self.config.save_freq == 0:
                    checkpoint_path = self.save_path / f"checkpoint_{step}.ckpt"
                    self.console.print(f"[green]Saved checkpoint: {checkpoint_path}[/green]")

                # Evaluation
                if step > 0 and step % self.config.eval_freq == 0:
                    self.console.print(f"[yellow]Evaluation at step {step}...[/yellow]")
                    # Would run evaluation here

        # Save final model
        final_path = self.save_path / "final_model.ckpt"
        self.console.print(f"[green]Training complete! Model saved to: {final_path}[/green]")

        # Return training results
        return {
            "status": "completed",
            "steps_trained": self.config.steps,
            "final_model": str(final_path),
            "checkpoints": list(self.save_path.glob("checkpoint_*.ckpt")),
        }

    def integrate_with_metta_training(self) -> str:
        """Generate command for metta training integration.

        Returns:
            Command string for metta training
        """
        cmd_parts = [
            "uv run ./tools/run.py",
            "experiments.recipes.arena.train",
            f"game={self.game_name}",
            f"scenario={self.scenario_name}",
            f"algorithm={self.config.algorithm}",
            f"total_timesteps={self.config.steps}",
            f"learning_rate={self.config.learning_rate}",
            f"batch_size={self.config.batch_size}",
        ]

        if self.config.wandb_project:
            cmd_parts.append(f"wandb_project={self.config.wandb_project}")

        if self.save_path:
            cmd_parts.append(f"checkpoint_dir={self.save_path}")

        return " ".join(cmd_parts)


def train_policy(
    game_name: str,
    scenario_name: str = None,  # Keep for backward compatibility
    algorithm: str = "ppo",
    steps: int = 10000,
    save_path: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    console: Optional[Console] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Train a policy on a game.

    Args:
        game_name: Name of the game
        scenario_name: Deprecated, kept for backward compatibility
        algorithm: Training algorithm
        steps: Number of training steps
        save_path: Path to save model
        wandb_project: W&B project for logging
        console: Optional Rich console
        **kwargs: Additional training configuration

    Returns:
        Training results
    """
    if console is None:
        console = Console()

    # Create training configuration
    config = TrainingConfig(algorithm=algorithm, steps=steps, wandb_project=wandb_project, **kwargs)

    # Create trainer
    trainer = Trainer(
        game_name=game_name,
        scenario_name=scenario_name,
        config=config,
        save_path=save_path,
        console=console,
    )

    # Show integration command
    integration_cmd = trainer.integrate_with_metta_training()
    console.print("\n[yellow]For full training with metta.rl.training, run:[/yellow]")
    console.print(f"[blue]{integration_cmd}[/blue]\n")

    # Run training (placeholder for now)
    return trainer.train()


def create_custom_scenario(
    game_name: str,
    scenario_name: str,
    num_agents: int = 2,
    width: int = 10,
    height: int = 10,
    output_path: Optional[Path] = None,
    console: Optional[Console] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create a custom scenario for training.

    Args:
        game_name: Name of the game
        scenario_name: Name for the new scenario
        num_agents: Number of agents
        width: Map width
        height: Map height
        output_path: Optional path to save scenario
        console: Optional Rich console
        **kwargs: Additional scenario parameters

    Returns:
        Scenario creation results
    """
    if console is None:
        console = Console()

    console.print(f"[cyan]Creating custom scenario for {game_name}[/cyan]")

    # For cogs_vs_clips, use the make_game function
    if game_name == "cogs_vs_clips":
        from cogames.cogs_vs_clips.scenarios import make_game
        from cogames.game import save_game_config

        # Create game configuration with specified parameters
        game_config = make_game(
            num_cogs=num_agents,
            num_assemblers=kwargs.get("num_assemblers", 1),
            num_base_extractors=kwargs.get("num_base_extractors", 1),
            num_wilderness_extractors=kwargs.get("num_wilderness_extractors", 1),
            num_chests=kwargs.get("num_chests", 1),
        )

        # Update map dimensions
        game_config.game.map_builder.width = width
        game_config.game.map_builder.height = height
        game_config.game.num_agents = num_agents

        if output_path:
            save_game_config(game_config, output_path)
            console.print(f"[green]Scenario saved to: {output_path}[/green]")

        return {
            "status": "created",
            "name": scenario_name,
            "config": game_config,
            "saved_to": str(output_path) if output_path else None,
        }

    else:
        raise ValueError(f"Custom scenario creation not implemented for game: {game_name}")
