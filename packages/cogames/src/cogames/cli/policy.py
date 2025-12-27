from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import typer
from pydantic import Field
from rich.table import Table

from cogames.cli.base import console
from mettagrid.policy.loader import resolve_policy_class_path
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.submission import POLICY_SPEC_FILENAME
from mettagrid.util.uri_resolvers.schemes import parse_uri, policy_spec_from_uri

RawPolicyValues = Optional[Sequence[str]]
ParsedPolicies = list[PolicySpec]

default_checkpoint_dir = Path("train_dir")

policy_arg_example = "class=NAME[,data=PATH][,proportion=1.0] or URI"
policy_arg_w_proportion_example = "class=NAME[,data=PATH][,proportion=1.0] or URI[,proportion=1.0]"


class PolicySpecWithProportion(PolicySpec):
    proportion: float = Field(default=1.0, description="Proportion of total agents to assign to this policy")

    def to_policy_spec(self) -> PolicySpec:
        return PolicySpec(
            class_path=self.class_path,
            data_path=self.data_path,
            init_kwargs=self.init_kwargs,
        )


def list_checkpoints():
    local_checkpoints = sorted(
        {path.parent for path in default_checkpoint_dir.rglob(POLICY_SPEC_FILENAME)},
        key=lambda path: path.stat().st_mtime,
    )
    if local_checkpoints:
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
    console.print("[bold]Class format[/bold] (policy class + optional data/kwargs):")
    console.print("  - class=random")
    console.print("  - class=random,data=./train_dir/run:v5/checkpoints/run:v5,proportion=0.5,kw.alpha=0.1")
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

    Supports two formats:
    - class=...[,data=...][,proportion=1.0][,kw.<key>=<value>]
    - URI: metta://policy/xxx[,proportion=1.0]
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")
    entries = [part.strip() for part in raw.split(",") if part.strip()]

    fraction = 1.0

    if not entries:
        raise ValueError("Policy specification cannot be empty.")

    if parse_uri(entries[0], allow_none=True, default_scheme=None):
        s = policy_spec_from_uri(entries[0])
        entries = entries[1:]

        for entry in entries:
            if "=" not in entry:
                raise ValueError(
                    "Policy entries must be key=value pairs "
                    "(e.g., class=stateless,data=train_dir/model.pt,proportion=0.5)."
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

    class_path: str | None = None
    data_path: str | None = None
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
                raise ValueError("Policy class cannot be empty.")
            class_path = resolve_policy_class_path(value)
            continue

        if key == "data":
            if not value:
                raise ValueError("Policy data path cannot be empty.")
            data_path = str(Path(value).expanduser().resolve())
            continue

        if key == "proportion":
            try:
                fraction = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid proportion value '{value}'.") from exc
            if fraction <= 0:
                raise ValueError("Policy proportion must be a positive number.")
            continue

        if key.startswith("kw."):
            kw_key = key[3:]
            if not kw_key:
                raise ValueError("Policy kw field name cannot be empty.")
            init_kwargs[kw_key.replace("-", "_")] = value
            continue

        raise ValueError(f"Unsupported policy field '{key}'.")

    if class_path is None:
        raise ValueError("Policy specification must include class= for key=value format.")

    return PolicySpecWithProportion(
        class_path=class_path,
        data_path=data_path,
        proportion=fraction,
        init_kwargs=init_kwargs,
    )
