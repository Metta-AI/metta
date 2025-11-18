import re
from collections import Counter
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS as CORE_EVAL_MISSIONS
from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS
from cogames.cogs_vs_clips.evals.spanning_evals import EVAL_MISSIONS as SPANNING_EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import MAP_MISSION_DELIMITER, Mission, MissionVariant, NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import SITES
from cogames.cogs_vs_clips.variants import HIDDEN_VARIANTS, VARIANTS
from cogames.game import load_mission_config, load_mission_config_from_python
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig
from mettagrid.mapgen.mapgen import MapGen

# Combined registry of all evaluation missions (not shown in default 'missions' list)
EVAL_MISSIONS_ALL: list[Mission] = [
    *CORE_EVAL_MISSIONS,
    *INTEGRATED_EVAL_MISSIONS,
    *SPANNING_EVAL_MISSIONS,
    *[mission_cls() for mission_cls in DIAGNOSTIC_EVALS],  # type: ignore[call-arg]
]


def _mission_name_counts(missions: list[Mission]) -> Counter[str]:
    """Count how many times each short mission name appears."""
    return Counter(mission.name for mission in missions)


def _dedupe_missions(missions: list[Mission]) -> list[Mission]:
    """Remove duplicate (site, mission) pairs while preserving order."""
    seen_keys: set[tuple[str, str]] = set()
    unique: list[Mission] = []
    for mission in missions:
        key = (mission.site.name, mission.name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(mission)
    return unique


def _canonical_mission_name(mission: Mission, *, counts: Counter[str]) -> str:
    """Prefer the short name unless it collides with another mission."""
    if counts[mission.name] == 1:
        return mission.name
    return mission.full_name()


def _mission_identifiers(
    missions: list[Mission],
    *,
    include_legacy: bool,
) -> list[str]:
    """Return deduplicated identifiers users can pass to the CLI.

    - Prefer short names when unique.
    - Keep `site.mission` names available for backwards compatibility.
    """
    unique_missions = _dedupe_missions(missions)
    counts = _mission_name_counts(unique_missions)

    identifiers: list[str] = []
    seen: set[str] = set()

    for mission in unique_missions:
        name = _canonical_mission_name(mission, counts=counts)
        if name not in seen:
            identifiers.append(name)
            seen.add(name)

    if include_legacy:
        for mission in unique_missions:
            legacy = mission.full_name()
            if legacy not in seen:
                identifiers.append(legacy)
                seen.add(legacy)

    return identifiers


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
        variant: Optional[MissionVariant] = None
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
    """Get all core mission names preferring short names when unique."""
    return _mission_identifiers(MISSIONS, include_legacy=False)


def get_all_eval_missions() -> list[str]:
    """Get all eval mission names preferring short names when unique."""
    return _mission_identifiers(EVAL_MISSIONS_ALL, include_legacy=False)


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
        available = _mission_identifiers([*MISSIONS, *EVAL_MISSIONS_ALL], include_legacy=True)
        missions = [m for m in available if re.search(regex_pattern, m)]
        # Drop the Mission (3rd element) for wildcard results
        return [
            (name, env_cfg)
            for name, env_cfg, _ in (get_mission(m, variants_arg=variants_arg, cogs=cogs) for m in missions)
        ]
    # Drop the Mission for single mission
    name, env_cfg, _ = get_mission(mission_arg, variants_arg=variants_arg, cogs=cogs)
    return [(name, env_cfg)]


def _find_mission(
    identifier: str,
    mission_name: Optional[str],
    *,
    include_evals: bool,
) -> Mission:
    """Resolve a mission by name or site-qualified name.

    Rules:
      - If mission_name is None and identifier matches a unique mission name, return it.
      - Otherwise treat identifier as a site name and optionally filter by mission_name.
      - If identifier is only a site name, the first mission on that site is returned
        to preserve previous behaviour.
    """
    mission_pool = _dedupe_missions([*MISSIONS, *(EVAL_MISSIONS_ALL if include_evals else [])])

    if mission_name is None and MAP_MISSION_DELIMITER not in identifier:
        name_matches = [mission for mission in mission_pool if mission.name == identifier]
        if len(name_matches) == 1:
            return name_matches[0]
        if len(name_matches) > 1:
            choices = ", ".join(sorted(m.full_name() for m in name_matches))
            raise ValueError(f"Mission name '{identifier}' is ambiguous. Try one of: {choices}")

    site_matches = [mission for mission in mission_pool if mission.site.name == identifier]
    if mission_name is None:
        if site_matches:
            return site_matches[0]
        raise ValueError(f"Could not find mission name or site {identifier}")

    filtered = [mission for mission in site_matches if mission.name == mission_name]
    if filtered:
        return filtered[0]

    if site_matches:
        raise ValueError(f"Mission {mission_name} not available on site {identifier}")
    raise ValueError(f"Could not find mission name or site {identifier}")


def find_mission(identifier: str, mission_name: Optional[str] = None) -> Mission:
    """Public helper that searches only the core mission catalog."""
    return _find_mission(identifier, mission_name, include_evals=False)


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

    if mission_arg.count(MAP_MISSION_DELIMITER) > 1:
        raise ValueError(f"Mission name can contain at most one `{MAP_MISSION_DELIMITER}` delimiter")

    if MAP_MISSION_DELIMITER in mission_arg:
        site_name, mission_name = mission_arg.split(MAP_MISSION_DELIMITER)
        mission = _find_mission(site_name, mission_name, include_evals=True)
    else:
        mission = _find_mission(mission_arg, None, include_evals=True)
    # Apply variants
    mission = mission.with_variants(variants)

    if cogs is not None:
        mission = mission.with_variants([NumCogsVariant(num_cogs=cogs)])

    canonical_name = _canonical_mission_name(
        mission,
        counts=_mission_name_counts(_dedupe_missions([*MISSIONS, *EVAL_MISSIONS_ALL])),
    )

    return (canonical_name, mission.make_env(), mission)


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

    missions = _dedupe_missions(MISSIONS)
    counts = _mission_name_counts(missions)

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")

    core_sites = [site for site in SITES if any(m.site.name == site.name for m in missions)]
    for idx, site in enumerate(core_sites):
        site_missions = [mission for mission in missions if mission.site.name == site.name]

        try:
            map_builder = site.map_builder
            if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                map_size = f"{map_builder.width}x{map_builder.height}"  # type: ignore[attr-defined]
            else:
                map_size = "N/A"
        except Exception:
            map_size = "N/A"

        agent_range = f"{site.min_cogs}-{site.max_cogs}"
        table.add_row(
            f"[bold white]{site.name}[/bold white]",
            agent_range,
            map_size,
            f"[dim]{site.description}[/dim]",
            end_section=True,
        )

        for mission_idx, mission in enumerate(site_missions):
            is_last_mission = mission_idx == len(site_missions) - 1
            is_last_site = idx == len(core_sites) - 1

            table.add_row(
                _canonical_mission_name(mission, counts=counts),
                "",
                "",
                mission.description,
            )

            if not is_last_mission:
                table.add_row("", "", "", "")
            elif not is_last_site:
                table.add_row("", "", "", "", end_section=True)

    console.print(table)

    console.print("\nTo specify a [bold blue] -m [MISSION][/bold blue], you can:")
    console.print("  • Use a mission name from above (e.g., [blue]harvest[/blue])")
    console.print("  • Use a path to a mission configuration file, e.g. path/to/mission.yaml")
    console.print("\nTo specify [bold yellow] -v [VARIANT][/bold yellow] modifiers:")
    console.print("  • Use multiple --variant flags: [yellow]--variant solar_flare --variant dark_side[/yellow]")
    console.print("  • Or use the short form: [yellow]-v solar_flare -v rough_terrain[/yellow]")
    console.print("\nTo specify number of cogs:")
    console.print("  • Use [green]--cogs N[/green] or [green]-c N[/green] (e.g., [green]-c 4[/green])")
    console.print("\n[bold green]Examples:[/bold green]")
    console.print("  [bold]cogames play[/bold] --mission [blue]harvest[/blue]")
    console.print(
        "  [bold]cogames play[/bold] --mission [blue]hello_world.open_world[/blue] --variant [yellow]mined_out[/yellow]"
    )
    console.print(
        "  [bold]cogames play[/bold] --mission [blue]machina_1.open_world[/blue] "
        "--variant [yellow]solar_flare[/yellow] --variant [yellow]rough_terrain[/yellow] "
        "--cogs [green]8[/green]"
    )
    console.print("  [bold]cogames train[/bold] --mission [blue]harvest[/blue] --cogs [green]4[/green]")


def list_evals() -> None:
    """Print a table listing all available eval missions."""
    if not EVAL_MISSIONS_ALL:
        console.print("No eval missions found")
        return

    missions_unique = _dedupe_missions(EVAL_MISSIONS_ALL)
    counts = _mission_name_counts(missions_unique)
    missions = sorted(
        missions_unique,
        key=lambda m: (m.site.name, _canonical_mission_name(m, counts=counts)),
    )

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
    table.add_column("Mission", style="blue", no_wrap=True)
    table.add_column("Site", style="white", no_wrap=True)
    table.add_column("Cogs", style="green", justify="center")
    table.add_column("Map Size", style="green", justify="center")
    table.add_column("Description", style="white")

    for mission in missions:
        site = mission.site
        try:
            map_builder = site.map_builder
            if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                map_size = f"{map_builder.width}x{map_builder.height}"  # type: ignore[attr-defined]
            else:
                map_size = "N/A"
        except Exception:
            map_size = "N/A"

        agent_range = f"{site.min_cogs}-{site.max_cogs}"
        table.add_row(
            _canonical_mission_name(mission, counts=counts),
            site.name,
            agent_range,
            map_size,
            mission.description,
        )

    console.print(table)
    console.print("\nTo play an eval mission:")
    console.print("  [bold]cogames play[/bold] --mission [blue]evals.oxygen_bottleneck[/blue]")


def describe_mission(mission_name: str, game_config: MettaGridConfig, mission_cfg: Optional[Mission] = None) -> None:
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
    console.print(f"  • Default resource limit: {game_config.game.agent.default_resource_limit}")
    if game_config.game.agent.resource_limits:
        console.print(f"  • Resource limits: {game_config.game.agent.resource_limits}")
