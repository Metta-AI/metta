"""CoGames CLI integration for Tribal Village."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from rich.console import Console

from tribal_village_env.cogames.train import train


def _import_cogames_deps():  # pragma: no cover - imported lazily for optional dependency
    from cogames.cli.base import console
    from cogames.cli.policy import get_policy_spec, policy_arg_example
    from cogames.device import resolve_training_device

    return console, get_policy_spec, policy_arg_example, resolve_training_device


def attach_train_command(
    app: typer.Typer,
    *,
    command_name: str = "train-tribal",
    require_cogames: bool = True,
    console_fallback: Optional[Console] = None,
) -> bool:
    try:
        console, get_policy_spec, policy_arg_example, resolve_training_device = _import_cogames_deps()
    except ImportError:
        if require_cogames:
            raise
        if console_fallback is not None:
            console_fallback.print("[yellow]CoGames not installed; Tribal train command unavailable.[/yellow]")
        return False

    @app.command(name=command_name, help="Train a policy on the Tribal Village environment")
    def train_tribal_cmd(  # noqa: PLR0913 - CLI surface mirrors cogames train
        ctx: typer.Context,
        policy: str = typer.Option("class=tribal", "--policy", "-p", help=f"Policy ({policy_arg_example})"),  # noqa: B008
        checkpoints_path: Path = typer.Option(  # noqa: B008
            Path("./train_dir"),
            "--checkpoints",
            help="Path to save training data",
        ),
        steps: int = typer.Option(10_000_000, "--steps", "-s", help="Number of training steps", min=1),  # noqa: B008
        device: str = typer.Option(  # noqa: B008
            "auto",
            "--device",
            help="Device to train on (e.g. 'auto', 'cpu', 'cuda')",
        ),
        seed: int = typer.Option(42, "--seed", help="Seed for training", min=0),  # noqa: B008
        batch_size: int = typer.Option(4096, "--batch-size", help="Batch size for training", min=1),  # noqa: B008
        minibatch_size: int = typer.Option(4096, "--minibatch-size", help="Minibatch size for training", min=1),  # noqa: B008
        num_workers: Optional[int] = typer.Option(  # noqa: B008
            None,
            "--num-workers",
            help="Number of worker processes (defaults to number of CPU cores)",
            min=1,
        ),
        parallel_envs: Optional[int] = typer.Option(  # noqa: B008
            64,
            "--parallel-envs",
            help="Number of parallel environments (defaults to 64 when omitted)",
            min=1,
        ),
        vector_batch_size: Optional[int] = typer.Option(  # noqa: B008
            None,
            "--vector-batch-size",
            help="Override vectorized environment batch size",
            min=1,
        ),
        max_steps: int = typer.Option(1000, "--episode-steps", help="Episode length", min=1),  # noqa: B008
        render_scale: int = typer.Option(  # noqa: B008
            1,
            "--render-scale",
            help="Scale factor for rendered frames (lower uses less memory)",
            min=1,
        ),
        render_mode: Literal["ansi", "rgb_array"] = typer.Option(  # noqa: B008
            "ansi",
            "--render-mode",
            help="Rendering mode used by the environment",
        ),
        log_outputs: bool = typer.Option(False, "--log-outputs", help="Log training outputs"),  # noqa: B008
    ) -> None:
        policy_spec = get_policy_spec(ctx, policy)
        torch_device = resolve_training_device(console, device)

        try:
            train(
                {
                    "policy_class_path": policy_spec.class_path,
                    "device": torch_device,
                    "checkpoints_path": checkpoints_path,
                    "steps": steps,
                    "seed": seed,
                    "batch_size": batch_size,
                    "minibatch_size": minibatch_size,
                    "vector_num_workers": num_workers,
                    "vector_num_envs": parallel_envs,
                    "vector_batch_size": vector_batch_size,
                    "env_config": {
                        "max_steps": max_steps,
                        "render_scale": render_scale,
                        "render_mode": render_mode,
                    },
                    "initial_weights_path": policy_spec.data_path,
                    "log_outputs": log_outputs,
                }
            )
        except ValueError as exc:  # pragma: no cover - user input
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")

    return True


def register_cli(app: typer.Typer) -> None:
    attach_train_command(app)


__all__ = ["attach_train_command", "register_cli"]
