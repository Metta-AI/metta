import re
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER, Mission, MissionVariant, NumCogsVariant, Site
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.cogs_vs_clips.sites import SITES
from cogames.cogs_vs_clips.variants import VARIANTS
from cogames.game import load_mission_config, load_mission_config_from_python
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig
from mettagrid.mapgen.mapgen import MapGen


def parse_variants(variants_arg: Optional[list[str]]) -> list[MissionVariant]:
    """Parse variant specifications from command line.

    Args:
        variants_arg: List of variant names like ["solar_flare", "dark_side"]

    Returns:
        List of configured MissionVariant instances

    Raises:
        ValueError: If variant name is unknown
    """
    if not variants_arg:
        return []

    variants: list[MissionVariant] = []
    for name in variants_arg:
        # Find matching variant class by instantiating and checking the name
        variant: MissionVariant | None = None
        for v in VARIANTS:
            if v.name == name:
                variant = v
                break

        if variant is None:
            # Get available variant names
            available = ", ".join(v.name for v in VARIANTS)
            raise ValueError(f"Unknown variant '{name}'.\nAvailable variants: {available}")

        # Instantiate with default configuration
        variants.append(variant)

    return variants


def get_all_missions() -> list[str]:
    """Get all mission names in the format site.mission."""
    return [mission.full_name() for mission in MISSIONS]


def get_site_by_name(site_name: str) -> Site:
    """Get a site by name.

    Raises:
        ValueError: If site not found
    """
    matching_sites = [site for site in SITES if site.name == site_name]
    if not matching_sites:
        raise ValueError(f"Could not find site {site_name}")
    elif len(matching_sites) > 1:
        raise ValueError(f"Invalid site catalog: more than one site named {site_name}")
    return matching_sites[0]


def get_mission_name_and_config(
    ctx: typer.Context, mission_arg: Optional[str], variants_arg: Optional[list[str]] = None, cogs: Optional[int] = None
) -> tuple[str, MettaGridConfig, Optional[Mission]]:
    if not mission_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --mission / -m[/yellow]\n")
    else:
        try:
            return get_mission(mission_arg, variants_arg, cogs)
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]\n")
    list_missions()

    if mission_arg is not None:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def get_mission_names_and_configs(
    ctx: typer.Context,
    missions_arg: Optional[list[str]],
    *,
    variants_arg: Optional[list[str]] = None,
    cogs: Optional[int] = None,
    steps: Optional[int] = None,
) -> list[tuple[str, MettaGridConfig]]:
    if not missions_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --mission / -m[/yellow]\n")
    else:
        try:
            not_deduped = [
                mission
                for missions in missions_arg
                for mission in _get_missions_by_possible_wildcard(missions, variants_arg, cogs)
            ]
            name_set: set[str] = set()
            deduped = []
            for m, c in not_deduped:
                if m not in name_set:
                    name_set.add(m)
                    deduped.append((m, c))
            if not deduped:
                raise ValueError(f"No missions found for {missions_arg}")

            # Apply steps override if explicitly provided
            if steps is not None:
                for _, env_cfg in deduped:
                    env_cfg.game.max_steps = steps

            return deduped
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]\n")
    list_missions()

    if missions_arg is not None:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def _get_missions_by_possible_wildcard(
    mission_arg: str,
    variants_arg: Optional[list[str]],
    cogs: Optional[int],
) -> list[tuple[str, MettaGridConfig]]:
    if "*" in mission_arg:
        # Convert shell-style wildcard to regex pattern
        regex_pattern = mission_arg.replace(".", "\\.").replace("*", ".*")
        missions = [m for m in get_all_missions() if re.search(regex_pattern, m)]
        # Drop the Mission (3rd element) for wildcard results
        return [
            (name, env_cfg)
            for name, env_cfg, _ in (get_mission(m, variants_arg=variants_arg, cogs=cogs) for m in missions)
        ]
    # Drop the Mission for single mission
    name, env_cfg, _ = get_mission(mission_arg, variants_arg=variants_arg, cogs=cogs)
    return [(name, env_cfg)]


def find_mission(
    site_name: str,
    mission_name: str | None = None,  # None means first mission on the site
) -> Mission:
    for mission in MISSIONS:
        if mission.site.name != site_name:
            continue
        if mission_name is not None and mission.name != mission_name:
            continue
        return mission

    if mission_name is not None:
        raise ValueError(f"Mission {mission_name} not available on site {site_name}")
    else:
        if site_name in [site.name for site in SITES]:
            raise ValueError(f"No missions available on site {site_name}")
        else:
            raise ValueError(f"Could not find mission name or site {site_name}")


def get_mission(
    mission_arg: str, variants_arg: Optional[list[str]] = None, cogs: Optional[int] = None
) -> tuple[str, MettaGridConfig, Optional[Mission]]:
    """Get a specific mission configuration by name or file path.

    Args:
        mission_arg: Name of the map or path to config file (.yaml, .json, or .py)
        variants_arg: List of variant names like ["solar_flare", "dark_side"]
        cogs: Number of cogs (agents) to use, overrides the default from the mission

    Returns:
        Tuple of (mission name, MettaGridConfig, Mission or None)

    Raises:
        ValueError: If mission not found or file cannot be loaded
    """
    # Check if it's a file path
    if any(mission_arg.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(mission_arg)
        if not path.exists():
            raise ValueError(f"File not found: {mission_arg}")
        if not path.is_file():
            raise ValueError(f"Not a file: {mission_arg}")

        # Load config based on file extension - no Mission available for file-based configs
        if path.suffix == ".py":
            return mission_arg, load_mission_config_from_python(path), None
        elif path.suffix in [".yaml", ".yml", ".json"]:
            return mission_arg, load_mission_config(path), None
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Parse variants if provided
    variants = parse_variants(variants_arg)

    # Otherwise, treat it as a fully qualified mission name, or as a site name
    if (delim_count := mission_arg.count(MAP_MISSION_DELIMITER)) == 0:
        site_name, mission_name = mission_arg, None
    elif delim_count > 1:
        raise ValueError(f"Mission name can contain at most one `{MAP_MISSION_DELIMITER}` delimiter")
    else:
        site_name, mission_name = mission_arg.split(MAP_MISSION_DELIMITER)

    mission = find_mission(site_name, mission_name)
    # Apply variants
    mission = mission.with_variants(variants)

    if cogs is not None:
        mission = mission.with_variants([NumCogsVariant(num_cogs=cogs)])

    return (
        mission.full_name(),
        mission.make_env(),
        mission,
    )


def list_variants() -> None:
    """Print a table listing all available variants."""
    if not VARIANTS:
        return

    console.print("\n")
    variant_table = Table(
        title="Available Variants", show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1)
    )
    variant_table.add_column("Variant", style="yellow", no_wrap=True)
    variant_table.add_column("Description", style="white")

    for variant in VARIANTS:
        variant_table.add_row(variant.name, variant.description)

    console.print(variant_table)


def list_missions() -> None:
    """Print a table listing all available missions."""

    if not SITES:
        console.print("No missions found")
        return

    # Create a single table for all missions
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")

    for idx, site in enumerate(SITES):
        # Get missions for this site
        site_missions = [mission for mission in MISSIONS if mission.site.name == site.name]

        # Get map size from the map builder
        try:
            map_builder = site.map_builder
            if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                map_size = f"{map_builder.width}x{map_builder.height}"  # type: ignore[attr-defined]
            else:
                map_size = "N/A"
        except Exception:
            map_size = "N/A"

        # Add site header row with map size and agent range
        agent_range = f"{site.min_cogs}-{site.max_cogs}"
        table.add_row(
            f"[bold white]{site.name}[/bold white]",
            agent_range,
            map_size,
            f"[dim]{site.description}[/dim]",
            end_section=True,
        )

        # Add missions for this site
        for mission_idx, mission in enumerate(site_missions):
            is_last_mission = mission_idx == len(site_missions) - 1
            is_last_site = idx == len(SITES) - 1

            # Add mission row with description in column
            table.add_row(
                mission.full_name(),
                "",
                "",
                mission.description,
            )

            # Add blank row for spacing between missions (except before section separator)
            if not is_last_mission:
                table.add_row("", "", "", "")
            elif not is_last_site:
                # Add separator after last mission if not the last site
                table.add_row("", "", "", "", end_section=True)

    console.print(table)

    # List variants in a separate table
    list_variants()

    console.print("\nTo specify a [bold blue] -m [MISSION][/bold blue], you can:")
    console.print("  • Use a mission name from above (e.g., [blue]training_facility.harvest[/blue])")
    console.print("  • Use a path to a mission configuration file, e.g. path/to/mission.yaml")
    console.print("\nTo specify [bold yellow] -v [VARIANT][/bold yellow] modifiers:")
    console.print("  • Use multiple --variant flags: [yellow]--variant solar_flare --variant dark_side[/yellow]")
    console.print("  • Or use the short form: [yellow]-v solar_flare -v rough_terrain[/yellow]")
    console.print("\nTo specify number of cogs:")
    console.print("  • Use [green]--cogs N[/green] or [green]-c N[/green] (e.g., [green]-c 4[/green])")
    console.print("\n[bold green]Examples:[/bold green]")
    console.print("  [bold]cogames play[/bold] --mission [blue]training_facility.harvest[/blue]")
    console.print(
        "  [bold]cogames play[/bold] --mission [blue]hello_world.explore[/blue] --variant [yellow]mined_out[/yellow]"
    )
    console.print(
        "  [bold]cogames play[/bold] --mission [blue]machina_1.open_world[/blue] "
        "--variant [yellow]solar_flare[/yellow] --variant [yellow]rough_terrain[/yellow] "
        "--cogs [green]8[/green]"
    )
    console.print(
        "  [bold]cogames train[/bold] --mission [blue]training_facility.harvest[/blue] --cogs [green]4[/green]"
    )


def describe_mission(mission_name: str, game_config: MettaGridConfig) -> None:
    """Print detailed information about a specific mission.

    Args:
        mission_name: Name of the mission
        env_cfg: Environment configuration
    """

    console.print(f"\n[bold cyan]{mission_name}[/bold cyan]\n")

    # Display mission configuration
    console.print("[bold]Mission Configuration:[/bold]")
    console.print(f"  • Number of agents: {game_config.game.num_agents}")
    if isinstance(game_config.game.map_builder, MapGen.Config):
        console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")

    # Display available actions
    console.print("\n[bold]Available Actions:[/bold]")
    for n, a in game_config.game.actions.model_dump().items():
        if a["enabled"]:
            console.print(f"  • {n}: {a['consumed_resources']}")

    # Display objects
    console.print("\n[bold]Stations:[/bold]")
    for obj_name, obj_config in game_config.game.objects.items():
        console.print(f"  • {obj_name}")
        if isinstance(obj_config, AssemblerConfig):
            for protocol in obj_config.protocols:
                if protocol.input_resources:
                    inputs = ", ".join(f"{k}:{v}" for k, v in protocol.input_resources.items())
                    outputs = ", ".join(f"{k}:{v}" for k, v in protocol.output_resources.items())
                    console.print(f"    {inputs} → {outputs} (cooldown: {protocol.cooldown})")

    # Display agent configuration
    console.print("\n[bold]Agent Configuration:[/bold]")
    console.print(f"  • Default resource limit: {game_config.game.agent.default_resource_limit}")
    if game_config.game.agent.resource_limits:
        console.print(f"  • Resource limits: {game_config.game.agent.resource_limits}")
