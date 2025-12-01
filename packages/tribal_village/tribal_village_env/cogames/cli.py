"""CoGames CLI integration for Tribal Village."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer

from cogames.cli.base import console
from cogames.cli.policy import get_policy_spec, policy_arg_example
from cogames.device import resolve_training_device

from . import policy as _policy  # noqa: F401  # Import for side effects (policy registration)
from . import train as tribal_train_module

__all__ = ["register_cli"]


def register_cli(app: typer.Typer) -> None:
    """Add Tribal Village commands to the CoGames Typer app."""

    @app.command(name="train-tribal", help="Train a policy on the Tribal Village environment")
    def train_tribal_cmd(
        ctx: typer.Context,
        policy: str = typer.Option("class=tribal", "--policy", "-p", help=f"Policy ({policy_arg_example})"),
        checkpoints_path: str = typer.Option(
            "./train_dir",
            "--checkpoints",
            help="Path to save training data",
        ),
        steps: int = typer.Option(10_000_000, "--steps", "-s", help="Number of training steps", min=1),
        device: str = typer.Option(
            "auto",
            "--device",
            help="Device to train on (e.g. 'auto', 'cpu', 'cuda')",
        ),
        seed: int = typer.Option(42, "--seed", help="Seed for training", min=0),
        batch_size: int = typer.Option(4096, "--batch-size", help="Batch size for training", min=1),
        minibatch_size: int = typer.Option(4096, "--minibatch-size", help="Minibatch size for training", min=1),
        num_workers: Optional[int] = typer.Option(
            None,
            "--num-workers",
            help="Number of worker processes (defaults to number of CPU cores)",
            min=1,
        ),
        parallel_envs: Optional[int] = typer.Option(
            64,
            "--parallel-envs",
            help="Number of parallel environments (defaults to 64 when omitted)",
            min=1,
        ),
        vector_batch_size: Optional[int] = typer.Option(
            None,
            "--vector-batch-size",
            help="Override vectorized environment batch size",
            min=1,
        ),
        max_steps: int = typer.Option(1000, "--episode-steps", help="Episode length for Tribal Village", min=1),
        render_scale: int = typer.Option(
            1,
            "--render-scale",
            help="Scale factor for rendered frames (lower uses less memory)",
            min=1,
        ),
        render_mode: Literal["ansi", "rgb_array"] = typer.Option(
            "ansi",
            "--render-mode",
            help="Rendering mode used by the environment",
        ),
        log_outputs: bool = typer.Option(False, "--log-outputs", help="Log training outputs"),
    ) -> None:
        policy_spec = get_policy_spec(ctx, policy)
        torch_device = resolve_training_device(console, device)

        env_config: dict[str, object] = {
            "max_steps": max_steps,
            "render_scale": render_scale,
            "render_mode": render_mode,
        }

        try:
            tribal_train_module.train(
                config=env_config,
                policy_class_path=policy_spec.class_path,
                initial_weights_path=policy_spec.data_path,
                device=torch_device,
                num_steps=steps,
                checkpoints_path=Path(checkpoints_path),
                seed=seed,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                vector_num_workers=num_workers,
                vector_num_envs=parallel_envs,
                vector_batch_size=vector_batch_size,
                log_outputs=log_outputs,
            )
        except ValueError as exc:  # pragma: no cover - user input
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")
