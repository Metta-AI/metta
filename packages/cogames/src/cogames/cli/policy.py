from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from rich.table import Table

from cogames.cli.base import console
from cogames.policy.interfaces import PolicySpec
from cogames.policy.utils import find_policy_checkpoints, resolve_policy_class_path, resolve_policy_data_path

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

_DELIMITER = ":"

policy_arg_w_proportion_example = _DELIMITER.join(
    (
        "[blue]CLASS[/blue]",
        "[cyan]DATA[/cyan]",
        "[light_slate_grey]:PROPORTION[/light_slate_grey]",
    )
)
policy_arg_example = _DELIMITER.join(
    (
        "[blue]CLASS[/blue]",
        "[cyan]DATA[/cyan]",
    )
)


def list_checkpoints():
    if local_checkpoints := find_policy_checkpoints(default_checkpoint_dir):
        table = Table(
            title="Local policy checkpoints usable as [cyan]DATA[/cyan]:", show_header=True, header_style="bold magenta"
        )

        table.add_column("Checkpoint", justify="right", style="bold cyan")
        table.add_column("Last modified", justify="right")

        for checkpoint in local_checkpoints:
            table.add_row(checkpoint.name, str(datetime.fromtimestamp(checkpoint.stat().st_mtime)))
        console.print(table)
        console.print()


def describe_policy_arg(with_proportion: bool):
    console.print(
        "To specify a [[bold cyan]POLICY[/bold cyan]], follow this format: "
        + (policy_arg_example if not with_proportion else policy_arg_w_proportion_example)
    )
    console.print()
    subcommand_parts = [
        "[blue]CLASS[/blue]: shorthand (e.g. 'simple', 'random') or fully qualified class path.",
        "[cyan]DATA[/cyan]: optional checkpoint path.",
    ]
    if with_proportion:
        subcommand_parts.append(
            "[light_slate_grey]PROPORTION[/light_slate_grey]: optional float specifying the population share."
        )
    console.print("\n".join([f"  - {part}" for part in subcommand_parts]))


def get_policy_spec(ctx: typer.Context, policy_arg: Optional[str]) -> PolicySpec:
    if policy_arg is None:
        console.print(ctx.get_help())
    else:
        try:
            return _parse_policy_spec(spec=policy_arg)  # type: ignore
        except ValueError as e:
            translated = str(e).replace("Invalid symbol name", "Could not find policy class")
            console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]")
            console.print()

    list_checkpoints()
    describe_policy_arg(with_proportion=False)
    console.print()
    if policy_arg is not None:
        console.print()
        console.print(ctx.get_usage())

    console.print()
    raise typer.Exit(0)


def get_policy_specs(ctx: typer.Context, policy_args: Optional[list[str]]) -> list[PolicySpec]:
    if not policy_args:
        console.print(ctx.get_help())
    else:
        try:
            return [_parse_policy_spec(spec=policy_arg) for policy_arg in policy_args]
        except ValueError as e:
            translated = str(e).replace("Invalid symbol name", "Could not find policy class")
            console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]")
            console.print()

    list_checkpoints()
    describe_policy_arg(with_proportion=True)
    console.print()
    if policy_args:
        console.print()
        console.print(ctx.get_usage())
    console.print()
    raise typer.Exit(0)


def _parse_policy_spec(
    spec: str,
    *,
    game_name: Optional[str] = None,
) -> PolicySpec:
    """Parse a policy CLI option into its components."""

    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = [part.strip() for part in raw.split(_DELIMITER)]
    if len(parts) > 3:
        raise ValueError(f"Policy specification must include at most two '{_DELIMITER}' separated values.")

    raw_class_path = parts[0]
    raw_policy_data = parts[1] if len(parts) > 1 else None
    raw_fraction = parts[2] if len(parts) > 2 else None

    if not raw_class_path:
        raise ValueError("Policy class path cannot be empty.")

    if not raw_fraction:
        fraction = 1.0
    else:
        try:
            fraction = float(raw_fraction)
        except ValueError as exc:
            raise ValueError(f"Invalid proportion value '{raw_fraction}'.") from exc

        if fraction <= 0:
            raise ValueError("Policy proportion must be a positive number.")

    resolved_class_path = resolve_policy_class_path(raw_class_path)
    resolved_policy_data = resolve_policy_data_path(
        raw_policy_data or None,
        policy_class_path=resolved_class_path,
        game_name=game_name,
    )

    return PolicySpec(
        policy_class_path=resolved_class_path,
        proportion=fraction,
        policy_data_path=resolved_policy_data,
    )
