from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from pydantic import Field
from rich.table import Table

from cogames.cli.base import console
from mettagrid.policy.loader import find_policy_checkpoints
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.uri_resolvers.schemes import parse_uri, policy_spec_from_uri

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

policy_arg_example = "URI"
policy_arg_w_proportion_example = "URI[,proportion=1.0]"


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
    console.print("[bold cyan]-p [POLICY][/bold cyan] accepts one format:\n")
    console.print("[bold]URI format[/bold] (checkpoint directory):")
    console.print("  - metta://policy/<name> or metta://policy/<uuid>")
    console.print("  - s3://bucket/path/to/checkpoints/run:v5")
    console.print("  - file:///path/to/checkpoints/run:v5 or /path/to/checkpoints/run:v5")
    if with_proportion:
        console.print(
            "  - [light_slate_grey]proportion[/light_slate_grey]: optional float specifying the population share.\n"
        )


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
    """Parse a policy CLI option into its components.

    Supports one format:
    - URI: metta://policy/xxx[,proportion=1.0]
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")
    entries = [part.strip() for part in raw.split(",") if part.strip()]

    fraction = 1.0

    if not entries or not parse_uri(entries[0], allow_none=True, default_scheme=None):
        raise ValueError("Policy specification must be a checkpoint URI.")

    s = policy_spec_from_uri(entries[0])
    entries = entries[1:]

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Policy entries must be key=value pairs (e.g., class=stateless,data=train_dir/model.pt,proportion=0.5)."
            )
        key, value = (part.strip() for part in entry.split("=", 1))

        if not key:
            raise ValueError("Policy field name cannot be empty.")

        if key == "proportion":
            try:
                fraction = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid proportion value '{value}'.") from exc
            if fraction <= 0:
                raise ValueError("Policy proportion must be a positive number.")
        else:
            raise ValueError("Only proportion is supported after a checkpoint URI.")

    return PolicySpecWithProportion(
        class_path=s.class_path,
        data_path=s.data_path,
        proportion=fraction,
        init_kwargs=s.init_kwargs,
    )
