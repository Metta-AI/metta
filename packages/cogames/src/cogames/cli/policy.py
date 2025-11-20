from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from rich.table import Table

from cogames.cli.base import console
from mettagrid.policy.loader import find_policy_checkpoints, resolve_policy_class_path, resolve_policy_data_path
from mettagrid.policy.policy import PolicySpec

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

POLICY_ARG_DELIMITER = ":"

policy_arg_example = POLICY_ARG_DELIMITER.join(
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


def describe_policy_arg():
    console.print("To specify a [bold cyan]-p [POLICY][/bold cyan], follow this format: " + policy_arg_example)
    subcommand_parts = [
        "[blue]CLASS[/blue]: shorthand (e.g. 'stateless', 'random') or fully qualified class path.",
        "[cyan]DATA[/cyan]: optional checkpoint path.",
    ]
    console.print("\n" + "\n".join([f"  - {part}" for part in subcommand_parts]) + "\n")


def _translate_error(e: Exception) -> str:
    translated = str(e).replace("Invalid symbol name", "Could not find policy class")
    if isinstance(e, ModuleNotFoundError):
        translated += ". Please make sure to specify your policy class."
    return translated


def get_policy_spec(ctx: typer.Context, policy_arg: Optional[str]) -> PolicySpec:
    if policy_arg is None:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --policy / -p[/yellow]\n")
    else:
        try:
            return _parse_policy_spec(spec=policy_arg)
        except (ValueError, ModuleNotFoundError) as e:
            translated = _translate_error(e)
            console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]\n")

    list_checkpoints()
    describe_policy_arg()

    if policy_arg is not None:
        console.print("\n" + ctx.get_usage())

    console.print("\n")
    raise typer.Exit(0)


def get_policy_specs(ctx: typer.Context, policy_args: Optional[list[str]]) -> list[PolicySpec]:
    if not policy_args:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --policy / -p[/yellow]\n")
    else:
        try:
            return [_parse_policy_spec(spec=policy_arg) for policy_arg in policy_args]
        except (ValueError, ModuleNotFoundError) as e:
            translated = _translate_error(e)
            console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]")
            console.print()

    list_checkpoints()
    describe_policy_arg()

    if policy_args:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def _parse_policy_spec(spec: str) -> PolicySpec:
    """Parse a policy CLI option into its components."""

    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    class_part, data_part = _split_class_and_data(raw)
    if not class_part:
        raise ValueError("Policy class path cannot be empty.")

    resolved_class_path = resolve_policy_class_path(class_part)
    resolved_policy_data = resolve_policy_data_path(data_part)

    return PolicySpec(
        class_path=resolved_class_path,
        data_path=resolved_policy_data,
    )


def _split_class_and_data(raw: str) -> tuple[str, Optional[str]]:
    if POLICY_ARG_DELIMITER not in raw:
        return raw.strip(), None

    class_part, data_part = raw.split(POLICY_ARG_DELIMITER, 1)
    data_part = data_part.strip() or None
    return class_part.strip(), data_part
