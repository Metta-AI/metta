import datetime
import pathlib
import typing

import rich.table
import typer

import cogames.cli.base
import mettagrid.policy.policy
import mettagrid.policy.utils

RawPolicyValues = typing.Optional[typing.Sequence[str]]
ParsedPolicies = list[mettagrid.policy.policy.PolicySpec]

default_checkpoint_dir = pathlib.Path("train_dir")

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


def list_checkpoints():
    if local_checkpoints := mettagrid.policy.utils.find_policy_checkpoints(default_checkpoint_dir):
        table = rich.table.Table(
            title="Local policy checkpoints usable as [cyan]DATA[/cyan]:", show_header=True, header_style="bold magenta"
        )

        table.add_column("Checkpoint", justify="right", style="bold cyan")
        table.add_column("Last modified", justify="right")

        for checkpoint in local_checkpoints:
            table.add_row(checkpoint.name, str(datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime)))
        cogames.cli.base.console.print(table)
        cogames.cli.base.console.print()


def describe_policy_arg(with_proportion: bool):
    cogames.cli.base.console.print(
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
    cogames.cli.base.console.print("\n" + "\n".join([f"  - {part}" for part in subcommand_parts]) + "\n")


def _translate_error(e: Exception) -> str:
    translated = str(e).replace("Invalid symbol name", "Could not find policy class")
    if isinstance(e, ModuleNotFoundError):
        translated += ". Please make sure to specify your policy class."
    return translated


def get_policy_spec(ctx: typer.Context, policy_arg: typing.Optional[str]) -> mettagrid.policy.policy.PolicySpec:
    if policy_arg is None:
        cogames.cli.base.console.print(ctx.get_help())
        cogames.cli.base.console.print("[yellow]Missing: --policy / -p[/yellow]\n")
    else:
        try:
            return _parse_policy_spec(spec=policy_arg)  # type: ignore
        except (ValueError, ModuleNotFoundError) as e:
            translated = _translate_error(e)
            cogames.cli.base.console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]\n")

    list_checkpoints()
    describe_policy_arg(with_proportion=False)

    if policy_arg is not None:
        cogames.cli.base.console.print("\n" + ctx.get_usage())

    cogames.cli.base.console.print("\n")
    raise typer.Exit(0)


def get_policy_specs(
    ctx: typer.Context, policy_args: typing.Optional[list[str]]
) -> list[mettagrid.policy.policy.PolicySpec]:
    if not policy_args:
        cogames.cli.base.console.print(ctx.get_help())
        cogames.cli.base.console.print("[yellow]Supply at least one: --policy / -p[/yellow]\n")
    else:
        try:
            return [_parse_policy_spec(spec=policy_arg) for policy_arg in policy_args]
        except (ValueError, ModuleNotFoundError) as e:
            translated = _translate_error(e)
            cogames.cli.base.console.print(f"[yellow]Error parsing policy argument: {translated}[/yellow]")
            cogames.cli.base.console.print()

    list_checkpoints()
    describe_policy_arg(with_proportion=True)

    if policy_args:
        cogames.cli.base.console.print("\n" + ctx.get_usage())
    cogames.cli.base.console.print("\n")
    raise typer.Exit(0)


def _parse_policy_spec(spec: str) -> mettagrid.policy.policy.PolicySpec:
    """Parse a policy CLI option into its components."""

    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = [part.strip() for part in raw.split(POLICY_ARG_DELIMITER)]
    if len(parts) > 3:
        raise ValueError(f"Policy specification must include at most two '{POLICY_ARG_DELIMITER}' separated values.")

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

    resolved_class_path = mettagrid.policy.utils.resolve_policy_class_path(raw_class_path)
    resolved_policy_data = mettagrid.policy.utils.resolve_policy_data_path(raw_policy_data or None)

    return mettagrid.policy.policy.PolicySpec(
        policy_class_path=resolved_class_path,
        proportion=fraction,
        policy_data_path=resolved_policy_data,
    )
