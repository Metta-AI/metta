from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from pydantic import Field
from rich.table import Table

from cogames.cli.base import console
from mettagrid.policy.loader import find_policy_checkpoints, resolve_policy_class_path, resolve_policy_data_path
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri, resolve_uri

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

policy_arg_example = "URI or class=CLS[,data=PATH][,kw.x=val]"
policy_arg_w_proportion_example = "URI or class=CLS[,data=PATH][,proportion=1.0][,kw.x=val]"


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
    console.print("[bold cyan]-p [POLICY][/bold cyan] accepts two formats:\n")
    console.print("[bold]1. URI format[/bold] (for .mpt checkpoints):")
    console.print("  - metta://policy/<name> or metta://policy/<uuid>")
    console.print("  - s3://bucket/path/to/checkpoint.mpt")
    console.print("  - file:///path/to/checkpoint.mpt or /path/to/checkpoint.mpt")
    console.print()
    console.print(
        "[bold]2. Key-value format[/bold]: "
        + (policy_arg_example if not with_proportion else policy_arg_w_proportion_example)
    )
    subcommand_parts = [
        "[blue]class[/blue]: shorthand (e.g. 'lstm', 'random') or fully qualified class path.",
        "[cyan]data[/cyan]: optional checkpoint path.",
    ]
    if with_proportion:
        subcommand_parts.append(
            "[light_slate_grey]proportion[/light_slate_grey]: optional float specifying the population share."
        )
    subcommand_parts.append("[magenta]kw.<arg>[/magenta]: optional policy __init__ kwarg (string values).")
    console.print("\n".join([f"  - {part}" for part in subcommand_parts]) + "\n")


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

    Supports two formats:
    - URI: metta://policy/xxx, s3://bucket/path, file:///path, or bare .mpt paths
    - Key-value: class=CLS[,data=PATH][,proportion=1.0][,kw.x=val]
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    try:
        resolved = resolve_uri(raw)
        if resolved.endswith(".mpt"):
            base_spec = policy_spec_from_uri(raw)
            return PolicySpecWithProportion(
                class_path=base_spec.class_path,
                init_kwargs=base_spec.init_kwargs,
                proportion=1.0,
            )
    except ValueError:
        pass
    except Exception as e:
        raise ValueError(f"Failed to resolve URI '{raw}': {e}") from e

    entries = [part.strip() for part in raw.split(",") if part.strip()]
    if not entries:
        raise ValueError(
            "Policy specification must use comma-separated key=value pairs "
            "(e.g., class=stateless,data=train_dir/model.pt,proportion=0.5)."
        )

    class_path: Optional[str] = None
    data_path: Optional[str] = None
    fraction = 1.0
    init_kwargs: dict[str, str] = {}

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Policy entries must be key=value pairs (e.g., class=stateless,data=train_dir/model.pt,proportion=0.5)."
            )
        key, value = (part.strip() for part in entry.split("=", 1))

        if not key:
            raise ValueError("Policy field name cannot be empty.")

        if key == "class":
            if not value:
                raise ValueError("Policy class path cannot be empty.")
            class_path = value
        elif key == "data":
            data_path = value or None
        elif key == "proportion":
            try:
                fraction = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid proportion value '{value}'.") from exc
            if fraction <= 0:
                raise ValueError("Policy proportion must be a positive number.")
        elif key.startswith("kw."):
            kw_key = key[3:]
            if not kw_key:
                raise ValueError("Policy kw.* entries must specify a name, e.g., kw.temperature=0.1.")
            init_kwargs[kw_key.replace("-", "_")] = value
        else:
            raise ValueError(f"Unknown policy field '{key}'. Expected class, data, proportion, or kw.<name> entries.")

    if class_path is None:
        raise ValueError("Policy specification must include a class entry (e.g., class=stateless).")

    resolved_class_path = resolve_policy_class_path(class_path)
    resolved_policy_data = resolve_policy_data_path(data_path) if data_path is not None else None

    return PolicySpecWithProportion(
        class_path=resolved_class_path,
        data_path=resolved_policy_data,
        proportion=fraction,
        init_kwargs=init_kwargs,
    )
