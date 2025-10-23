import re
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.missions import MISSIONS, SITES, VARIANTS
from cogames.game import load_mission_config, load_mission_config_from_python
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig

MAP_MISSION_DELIMITER = "."


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
        variant_class = None
        for v_class in VARIANTS:
            # Create temporary instance to check the name
            temp_instance = v_class()
            if temp_instance.name == name or v_class.__name__.lower().replace("variant", "") == name:
                variant_class = v_class
                break

        if variant_class is None:
            # Get available variant names
            available = ", ".join(v_class().name for v_class in VARIANTS)
            raise ValueError(f"Unknown variant '{name}'.\nAvailable variants: {available}")

        # Instantiate with default configuration
        variants.append(variant_class())

    return variants


def get_all_missions() -> list[str]:
    """Get all mission names in the format site.mission."""
    return [f"{mission_class().site.name}{MAP_MISSION_DELIMITER}{mission_class().name}" for mission_class in MISSIONS]


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
    ctx: typer.Context, missions_arg: Optional[list[str]]
) -> list[tuple[str, MettaGridConfig]]:
    if not missions_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --mission / -m[/yellow]\n")
    else:
        try:
            not_deduped = [
                mission for missions in missions_arg for mission in _get_missions_by_possible_wildcard(missions)
            ]
            name_set: set[str] = set()
            deduped = []
            for m, c in not_deduped:
                if m not in name_set:
                    name_set.add(m)
                    deduped.append((m, c))
            if not deduped:
                raise ValueError(f"No missions found for {missions_arg}")
            return deduped
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]\n")
    list_missions()

    if missions_arg is not None:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def _get_missions_by_possible_wildcard(mission_arg: str) -> list[tuple[str, MettaGridConfig]]:
    if "*" in mission_arg:
        # Convert shell-style wildcard to regex pattern
        regex_pattern = mission_arg.replace(".", "\\.").replace("*", ".*")
        missions = [m for m in get_all_missions() if re.search(regex_pattern, m)]
        # Drop the Mission (3rd element) for wildcard results
        return [(name, env_cfg) for name, env_cfg, _ in (get_mission(m) for m in missions)]
    # Drop the Mission for single mission
    name, env_cfg, _ = get_mission(mission_arg)
    return [(name, env_cfg)]


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

    # Otherwise, treat it as a mission name
    if (delim_count := mission_arg.count(MAP_MISSION_DELIMITER)) == 0:
        site_name, mission_name = mission_arg, None
    elif delim_count > 1:
        raise ValueError(f"Mission name can contain at most one `{MAP_MISSION_DELIMITER}` delimiter")
    else:
        site_name, mission_name = mission_arg.split(MAP_MISSION_DELIMITER)

    # Get the site
    site = get_site_by_name(site_name)

    # Determine number of cogs to use
    num_cogs = cogs if cogs is not None else site.min_cogs

    # Apply variants to the mission
    def apply_variants_to_config(
        mission: Mission, map_builder: MapBuilderConfig, num_cogs: int
    ) -> tuple[MettaGridConfig, Mission]:
        """Apply variants and return both MettaGridConfig and Mission.

        Important: Variants must be applied BEFORE finalizing the map builder.
        For procedural missions, the map builder is reconstructed inside
        `instantiate` based on `procedural_overrides`. If we apply variants
        after instantiation, those overrides won't affect the generated map.
        To preserve existing variant semantics (multiple variants in order and
        after `configure()`), we compose the provided variants into a single
        variant and pass it into `instantiate`.
        """

        if variants:
            # Compose multiple variants so they apply in order during instantiate
            class _CombinedVariant(MissionVariant):
                name: str = "combined"
                description: str = "Composite of CLI variants applied in order"

                def apply(self, m: Mission) -> Mission:  # type: ignore[override]
                    for v in variants:
                        m = v.apply(m)
                    return m

            combined_variant = _CombinedVariant()
        else:
            combined_variant = None

        # Instantiate with the combined variant to ensure overrides affect map
        instantiated_mission = mission.instantiate(map_builder, num_cogs, combined_variant)
        return instantiated_mission.make_env(), instantiated_mission

    if mission_name is not None:
        # Find the matching mission class
        matching_mission_class = None
        for mission_class in MISSIONS:
            temp_mission = mission_class()
            if temp_mission.site.name == site_name and temp_mission.name == mission_name:
                matching_mission_class = mission_class
                break

        if not matching_mission_class:
            raise ValueError(f"Mission {mission_name} not available on site {site_name}")

        mission_instance = matching_mission_class()
        env_cfg, mission_cfg = apply_variants_to_config(mission_instance, site.map_builder, num_cogs)
        return (
            f"{site.name}{MAP_MISSION_DELIMITER}{mission_name}",
            env_cfg,
            mission_cfg,
        )
    else:
        # Use first mission of the site if no specific mission is specified
        site_missions = [m for m in MISSIONS if m().site.name == site_name]
        if site_missions:
            first_mission_class = site_missions[0]
            first_mission = first_mission_class()
            env_cfg, mission_cfg = apply_variants_to_config(first_mission, site.map_builder, num_cogs)
            return (
                f"{site.name}{MAP_MISSION_DELIMITER}{first_mission.name}",
                env_cfg,
                mission_cfg,
            )
        else:
            # Fallback to default map if no missions exist
            built_map = site.map_builder.create().build()
            return f"{site.name}", built_map.make_env(), None


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

    for variant_class in VARIANTS:
        variant_instance = variant_class()
        variant_table.add_row(variant_instance.name, variant_instance.description)

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
        site_missions = [mission_class for mission_class in MISSIONS if mission_class().site.name == site.name]

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
        for mission_idx, mission_class in enumerate(site_missions):
            mission_instance = mission_class()
            mission_name = f"{site.name}.{mission_instance.name}"
            is_last_mission = mission_idx == len(site_missions) - 1
            is_last_site = idx == len(SITES) - 1

            # Add mission row with description in column
            table.add_row(
                mission_name,
                "",
                "",
                mission_instance.description,
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
    console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")  # type: ignore[attr-defined]

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
            for _, recipe in obj_config.recipes:
                if recipe.input_resources:
                    inputs = ", ".join(f"{k}:{v}" for k, v in recipe.input_resources.items())
                    outputs = ", ".join(f"{k}:{v}" for k, v in recipe.output_resources.items())
                    console.print(f"    {inputs} → {outputs} (cooldown: {recipe.cooldown})")

    # Display agent configuration
    console.print("\n[bold]Agent Configuration:[/bold]")
    console.print(f"  • Default resource limit: {game_config.game.agent.default_resource_limit}")
    if game_config.game.agent.resource_limits:
        console.print(f"  • Resource limits: {game_config.game.agent.resource_limits}")
