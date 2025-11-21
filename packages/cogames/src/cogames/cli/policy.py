from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from pydantic import Field
from rich.table import Table

from cogames.cli.base import console
from metta.rl.checkpoint_manager import CheckpointManager
from mettagrid.policy.loader import find_policy_checkpoints, resolve_policy_class_path, resolve_policy_data_path
from mettagrid.policy.policy import PolicySpec

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

POLICY_ARG_DELIMITER = ":"

policy_arg_w_proportion_example = POLICY_ARG_DELIMITER.join(
    (
        "[blue]CLASS[/blue]",
        "[cyan]DATA[/cyan]",
        "[light_slate_grey]:PROPORTION[/light_slate_grey]",
    )
)
policy_arg_example = POLICY_ARG_DELIMITER.join(
    (
        "[blue]CLASS[/blue]",
        "[cyan]DATA[/cyan]",
    )
)


class PolicySpecWithProportion(PolicySpec):
    proportion: float = Field(default=1.0, description="Proportion of total agents to assign to this policy")

    def to_policy_spec(self) -> PolicySpec:
        return PolicySpec(
            class_path=self.class_path,
            data_path=self.data_path,
            init_kwargs=self.init_kwargs,
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
        "To specify a [bold cyan]-p [POLICY][/bold cyan], follow this format: "
        + (policy_arg_example if not with_proportion else policy_arg_w_proportion_example)
    )
    subcommand_parts = [
        "[blue]CLASS[/blue]: shorthand (e.g. 'stateless', 'random') or fully qualified class path.",
        "[cyan]DATA[/cyan]: optional checkpoint path.",
    ]
    if with_proportion:
        subcommand_parts.append(
            "[light_slate_grey]PROPORTION[/light_slate_grey]: optional float specifying the population share."
        )
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
            return _parse_policy_spec(spec=policy_arg).to_policy_spec()
        except (ValueError, ModuleNotFoundError) as e:
            translated = _translate_error(e)
            console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]\n")

    list_checkpoints()
    describe_policy_arg(with_proportion=False)

    if policy_arg is not None:
        console.print("\n" + ctx.get_usage())

    console.print("\n")
    raise typer.Exit(0)


def get_policy_specs_with_proportions(
    ctx: typer.Context, policy_args: Optional[list[str]]
) -> list[PolicySpecWithProportion]:
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
    describe_policy_arg(with_proportion=True)

    if policy_args:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def _parse_policy_spec(spec: str) -> PolicySpecWithProportion:
    """Parse a policy CLI option into its components."""

    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    fraction = 1.0
    body = raw
    head, sep, tail = raw.rpartition(POLICY_ARG_DELIMITER)
    if sep:
        candidate = tail.strip()
        if candidate:
            try:
                parsed_fraction = float(candidate)
            except ValueError:
                pass
            else:
                if parsed_fraction <= 0:
                    raise ValueError("Policy proportion must be a positive number.")
                fraction = parsed_fraction
                body = head

    if not body:
        raise ValueError("Policy specification missing class path.")

    if POLICY_ARG_DELIMITER in body:
        class_part, data_part_raw = body.split(POLICY_ARG_DELIMITER, 1)
        data_part = data_part_raw.strip() or None
    else:
        class_part = body
        data_part = None

    class_part = class_part.strip()
    if not class_part:
        raise ValueError("Policy class path cannot be empty.")

    resolved_class_path = resolve_policy_class_path(class_part)
    resolved_policy_data = resolve_policy_data_path(data_part)

    artifact_spec = _maybe_policy_artifact_spec(
        display_name=class_part,
        checkpoint_path=resolved_policy_data,
        proportion=fraction,
    )
    if artifact_spec is not None:
        return artifact_spec

    return PolicySpecWithProportion(
        class_path=resolved_class_path,
        data_path=resolved_policy_data,
        proportion=fraction,
    )


def _maybe_policy_artifact_spec(
    *,
    display_name: str,
    checkpoint_path: Optional[str],
    proportion: float,
) -> PolicySpecWithProportion | None:
    if not checkpoint_path or not checkpoint_path.lower().endswith(".mpt"):
        return None

    spec = CheckpointManager.policy_spec_from_uri(checkpoint_path, display_name=display_name)

    return PolicySpecWithProportion(
        class_path=spec.class_path,
        data_path=spec.data_path,
        init_kwargs=spec.init_kwargs,
        proportion=proportion,
    )
