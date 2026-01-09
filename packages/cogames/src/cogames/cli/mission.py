import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER, Mission, MissionVariant, NumCogsVariant, Site
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import SITES
from cogames.cogs_vs_clips.variants import HIDDEN_VARIANTS, VARIANTS
from cogames.game import load_mission_config, load_mission_config_from_python
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig
from mettagrid.mapgen.mapgen import MapGen


@lru_cache(maxsize=1)
def _get_core_missions() -> list[Mission]:
    from cogames.cogs_vs_clips.missions import get_core_missions

    return get_core_missions()


@lru_cache(maxsize=1)
def _get_eval_missions_all() -> list[Mission]:
    from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
    from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS
    from cogames.cogs_vs_clips.evals.spanning_evals import EVAL_MISSIONS as SPANNING_EVAL_MISSIONS

    missions: list[Mission] = []
    missions.extend(INTEGRATED_EVAL_MISSIONS)
    missions.extend(SPANNING_EVAL_MISSIONS)
    missions.extend(mission_cls() for mission_cls in DIAGNOSTIC_EVALS)  # type: ignore[call-arg]
    return missions


def load_mission_set(mission_set: str) -> list[Mission]:
    """Load a predefined set of evaluation missions.

    Args:
        mission_set: Name of mission set to load. Options:
            - "integrated_evals": Integrated evaluation missions
            - "spanning_evals": Spanning evaluation missions
            - "diagnostic_evals": Diagnostic evaluation missions
            - "all": All missions including core missions

    Returns:
        List of Mission objects in the specified set

    Raises:
        ValueError: If mission_set name is unknown
    """
    if mission_set == "all":
        # All missions: eval missions + integrated + spanning + diagnostic + core missions
        missions_list = list(_get_eval_missions_all())

        # Add core missions that aren't already in eval sets
        eval_mission_names = {m.name for m in missions_list}
        for mission in _get_core_missions():
            if mission.name not in eval_mission_names:
                missions_list.append(mission)

    elif mission_set == "diagnostic_evals":
        from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS

        missions_list: list[Mission] = [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]  # type: ignore[call-arg]
    elif mission_set == "integrated_evals":
        from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS

        missions_list = list(INTEGRATED_EVAL_MISSIONS)
    elif mission_set == "spanning_evals":
        from cogames.cogs_vs_clips.evals.spanning_evals import EVAL_MISSIONS as SPANNING_EVAL_MISSIONS

        missions_list = list(SPANNING_EVAL_MISSIONS)
    else:
        available = "eval_missions, integrated_evals, spanning_evals, diagnostic_evals, all"
        raise ValueError(f"Unknown mission set: {mission_set}\nAvailable sets: {available}")

    return missions_list


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
    all_variants = [*VARIANTS, *HIDDEN_VARIANTS]
    for name in variants_arg:
        # Find matching variant class by instantiating and checking the name
        variant: MissionVariant | None = None
        for v in all_variants:
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
    """Get all core mission names in the format site.mission (excludes evals)."""
    return [mission.full_name() for mission in _get_core_missions()]


def get_all_eval_missions() -> list[str]:
    """Get all eval mission names in the format site.mission."""
    return [mission.full_name() for mission in _get_eval_missions_all()]


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
        missions = [m for m in (get_all_missions() + get_all_eval_missions()) if re.search(regex_pattern, m)]
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
    *,
    include_evals: bool = False,
) -> Mission:
    missions = _get_core_missions()
    if include_evals:
        missions = [*missions, *_get_eval_missions_all()]

    found_site = False
    for mission in missions:
        if mission.site.name != site_name:
            continue
        found_site = True
        if mission_name is not None and mission.name != mission_name:
            continue
        return mission

    if mission_name is None and not found_site:
        for mission in missions:
            if mission.name == site_name:
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

    mission = find_mission(site_name, mission_name, include_evals=True)
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


def list_missions(site_filter: Optional[str] = None) -> None:
    """List missions: sites only by default; expand sub-missions when a site is provided."""

    if not SITES:
        console.print("No missions found")
        return

    normalized_filter = site_filter.rstrip(".") if site_filter is not None else None
    core_missions = _get_core_missions()

    # Create a single table for all missions
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")

    core_sites = [site for site in SITES if any(m.site.name == site.name for m in core_missions)]

    if normalized_filter is not None:
        core_sites = [site for site in core_sites if site.name == normalized_filter]
        if not core_sites:
            console.print(f"[red]No missions found for site '{normalized_filter}'[/red]")
            return

    for site in core_sites:
        # Get missions for this site
        site_missions = [mission for mission in core_missions if mission.site.name == site.name]

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
            end_section=normalized_filter is None,
        )

        if normalized_filter is None:
            continue

        # Add missions for this site
        for mission_idx, mission in enumerate(site_missions):
            is_last_mission = mission_idx == len(site_missions) - 1

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

    console.print(table)

    console.print("\nTo set [bold blue]-m[/bold blue]:")
    console.print("  • Use [blue]<site>.<mission>[/blue] (e.g., training_facility.harvest)")
    console.print("  • Or pass a mission config file path")
    console.print("  • List a site's missions: [blue]cogames missions training_facility[/blue]")
    console.print("\nVariants:")
    console.print("  • Repeat [yellow]--variant <name>[/yellow] (e.g., --variant solar_flare)")
    console.print("\nCogs:")
    console.print("  • [green]--cogs N[/green] or [green]-c N[/green]")
    console.print("\n[bold green]Examples:[/bold green]")
    console.print("  cogames missions")
    console.print("  cogames missions training_facility")
    console.print("  cogames play --mission [blue]training_facility.harvest[/blue]")
    console.print(
        "  cogames play --mission [blue]machina_1.open_world[/blue] "
        "--variant [yellow]solar_flare[/yellow] --variant [yellow]rough_terrain[/yellow] --cogs [green]8[/green]"
    )
    console.print("  cogames train --mission [blue]<site>.<mission>[/blue] --cogs [green]4[/green]")


def list_evals() -> None:
    """Print a table listing all available eval missions."""
    evals = _get_eval_missions_all()
    if not evals:
        console.print("No eval missions found")
        return

    # Group missions by site
    missions_by_site: dict[str, list[Mission]] = {}
    for m in evals:
        missions_by_site.setdefault(m.site.name, []).append(m)

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")

    site_names = sorted(missions_by_site.keys())
    for idx, site_name in enumerate(site_names):
        site = next((s for s in SITES if s.name == site_name), None)
        # Determine map size if possible
        try:
            if site is not None and hasattr(site.map_builder, "width") and hasattr(site.map_builder, "height"):
                map_size = f"{site.map_builder.width}x{site.map_builder.height}"  # type: ignore[attr-defined]
            else:
                map_size = "N/A"
        except Exception:
            map_size = "N/A"

        agent_range = f"{site.min_cogs}-{site.max_cogs}" if site is not None else ""
        description = site.description if site is not None else ""
        table.add_row(
            f"[bold white]{site_name}[/bold white]",
            agent_range,
            map_size,
            f"[dim]{description}[/dim]",
            end_section=True,
        )

        site_missions = missions_by_site[site_name]
        for mission_idx, mission in enumerate(site_missions):
            is_last_mission = mission_idx == len(site_missions) - 1
            is_last_site = idx == len(site_names) - 1
            table.add_row(
                mission.full_name(),
                "",
                "",
                mission.description,
            )
            if not is_last_mission:
                table.add_row("", "", "", "")
            elif not is_last_site:
                table.add_row("", "", "", "", end_section=True)

    console.print(table)
    console.print("\nTo play an eval mission:")
    console.print("  [bold]cogames play[/bold] --mission [blue]evals.divide_and_conquer[/blue]")


def describe_mission(mission_name: str, game_config: MettaGridConfig, mission_cfg: Mission | None = None) -> None:
    """Print detailed information about a specific mission.

    Args:
        mission_name: Name of the mission
        game_config: Environment configuration
        mission_cfg: Mission object if available (to show description and variants)
    """

    console.print(f"\n[bold cyan]{mission_name}[/bold cyan]\n")

    if mission_cfg is not None:
        # Human-facing mission description
        console.print("[bold]Description:[/bold]")
        console.print(f"  {mission_cfg.description}\n")

        # Variants applied
        if mission_cfg.variants:
            console.print("[bold]Variants Applied:[/bold]")
            for v in mission_cfg.variants:
                desc = f" - {v.description}" if getattr(v, "description", "") else ""
                console.print(f"  • {v.name}{desc}")
            console.print("")

    # Display mission configuration
    console.print("[bold]Mission Configuration:[/bold]")
    console.print(f"  • Number of agents: {game_config.game.num_agents}")
    if isinstance(game_config.game.map_builder, MapGen.Config):
        console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")
        # Show procedural map details (e.g., biome from variants like -v desert)
        instance = getattr(game_config.game.map_builder, "instance", None)
        if isinstance(instance, MachinaArena.Config):
            console.print("\n[bold]MapGen (MachinaArena):[/bold]")
            console.print(f"  • Base biome: {instance.base_biome}")
            if instance.biome_weights:
                console.print(f"  • Biome weights: {instance.biome_weights}")
            console.print(f"  • Building coverage: {instance.building_coverage}")
    # Key knobs
    console.print(
        f"  • Regen interval: {game_config.game.inventory_regen_interval}, "
        f"Move energy cost: {game_config.game.actions.move.consumed_resources.get('energy', 0)}"
    )
    # Clipping info
    clip_period = getattr(game_config.game.clipper, "clip_period", 0)
    if clip_period:
        console.print(f"  • Clip period: {clip_period}")

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
    console.print(f"  • Default resource limit: {game_config.game.agent.inventory.default_limit}")
    if game_config.game.agent.inventory.limits:
        console.print(f"  • Resource limits: {game_config.game.agent.inventory.limits}")
