"""
Cogs vs Clips environment configuration and setup.

This module provides the main function to create a complete Cogs vs Clips game environment
with all the game mechanics, objects, and maps configured according to the game rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from mettagrid.src.metta.mettagrid.mettagrid_config import (
    GameConfig, AgentConfig, ConverterConfig, GroupConverterConfig, GroupConverterRecipe, Direction
)

# Resource type IDs
RESOURCE_IDS = {
    "rare_earth": 0,
    "trapped_helium": 1,
    "mercury": 2,
    "battery": 3,
    "heart": 4,
}

# Object type IDs
OBJECT_IDS = {
    "nano_assembler": 20,
    "rare_earth_depleted": 21,
    "trapped_helium_depleted": 22,
    "mercury_depleted": 23,
    "battery_solar": 24,
    "rare_earth_rich": 31,
    "trapped_helium_rich": 32,
    "mercury_rich": 33,
    "battery_rich": 34,
    "rare_earth_infected": 41,
    "trapped_helium_infected": 42,
    "mercury_infected": 43,
    "battery_infected": 44,
    "hearts_chest": 8,
    "rare_earth_depot": 9,
    "trapped_helium_depot": 10,
    "mercury_depot": 11,
    "battery_depot": 12,
}

@dataclass
class CogsVsClipsConfig:
    """Configuration parameters for Cogs vs Clips game."""
    grid_size: Tuple[int, int] = (40, 40)
    max_agents: int = 20
    max_steps: int = 10000

    # Resource limits
    agent_resource_capacity: int = 50
    agent_heart_capacity: int = 1

    # Infection parameters
    infection_base_rate: float = 0.01
    infection_growth_rate: float = 1.1
    infection_period: float = 100.0
    infection_proximity_boost: float = 2.0

    # Recipe requirements
    heart_recipe_cost: int = 10  # of each resource
    tool_recipe_costs: Dict[str, Dict[str, int]] = None

    def __post_init__(self):
        if self.tool_recipe_costs is None:
            self.tool_recipe_costs = {
                "nano_disruptor": {"rare_earth": 5, "battery": 3},
                "magnetic_resonator": {"trapped_helium": 5, "battery": 2},
                "quantum_modulator": {"mercury": 3, "battery": 5},
                "laser": {"rare_earth": 2, "trapped_helium": 3, "mercury": 2},
            }


def create_nano_assembler_config() -> GroupConverterConfig:
    """Create the nano-assembler with all recipes configured using helper methods."""
    config = GroupConverterConfig(
        type_id=OBJECT_IDS["nano_assembler"],
        conversion_ticks=20,
        cooldown=10,
        color=1
    )

    # Heart recipe - requires agents at N, E, S, W stations
    pattern, recipe = GroupConverterRecipe.make(
        recipe=["N", "E", "S", "W"],
        consumes=[("rare_earth", 10), ("trapped_helium", 10), ("mercury", 10)],
        produces=[("heart", 1)]
    )
    config.recipes[pattern] = recipe

    # Tool recipes
    # Nano Disruptor - requires agent at North station
    pattern, recipe = GroupConverterRecipe.make(
        recipe=["N"],
        consumes=[("rare_earth", 5), ("battery", 3)],
        produces=[("nano_disruptor", 1)]
    )
    config.recipes[pattern] = recipe

    # Magnetic Resonator - requires agent at East station
    pattern, recipe = GroupConverterRecipe.make(
        recipe=["E"],
        consumes=[("trapped_helium", 5), ("battery", 2)],
        produces=[("magnetic_resonator", 1)]
    )
    config.recipes[pattern] = recipe

    # Quantum Modulator - requires agent at West station
    pattern, recipe = GroupConverterRecipe.make(
        recipe=["W"],
        consumes=[("mercury", 3), ("battery", 5)],
        produces=[("quantum_modulator", 1)]
    )
    config.recipes[pattern] = recipe

    # Laser - requires agent at South station
    pattern, recipe = GroupConverterRecipe.make(
        recipe=["S"],
        consumes=[("rare_earth", 2), ("trapped_helium", 3), ("mercury", 2)],
        produces=[("laser", 1)]
    )
    config.recipes[pattern] = recipe

    return config


def create_resource_extractors() -> Dict[int, ConverterConfig]:
    """Create all resource extractor configurations."""
    extractors = {}

    # Depleted resource patches (low yield)
    for resource, resource_id in RESOURCE_IDS.items():
        if resource == "heart":  # Skip heart, it's manufactured
            continue

        if resource == "battery":
            # Solar stations generate batteries
            extractors[OBJECT_IDS["battery_solar"]] = ConverterConfig(
                type_id=OBJECT_IDS["battery_solar"],
                input_resources={},  # No input required
                output_resources={"battery": 1},
                max_output=10,
                max_conversions=-1,
                conversion_ticks=30,
                cooldown=20,
                color=3
            )
        else:
            # Regular resource patches
            depleted_id = OBJECT_IDS[f"{resource}_depleted"]
            extractors[depleted_id] = ConverterConfig(
                type_id=depleted_id,
                input_resources={},
                output_resources={resource: 1},
                max_output=5,
                max_conversions=50,  # Limited extractions
                conversion_ticks=10,
                cooldown=15,
                initial_resource_count=2,
                color=2
            )

            # Rich resource patches (high yield)
            rich_id = OBJECT_IDS[f"{resource}_rich"]
            extractors[rich_id] = ConverterConfig(
                type_id=rich_id,
                input_resources={},
                output_resources={resource: 3},
                max_output=15,
                max_conversions=100,
                conversion_ticks=5,
                cooldown=10,
                initial_resource_count=5,
                color=4
            )

            # Infected patches (produce paperclips instead)
            infected_id = OBJECT_IDS[f"{resource}_infected"]
            extractors[infected_id] = ConverterConfig(
                type_id=infected_id,
                input_resources={},
                output_resources={"paperclip": 1},  # Infected output
                max_output=20,
                max_conversions=-1,  # Unlimited infection spread
                conversion_ticks=8,
                cooldown=5,
                color=5
            )

    return extractors


def create_storage_facilities() -> Dict[int, ConverterConfig]:
    """Create storage depot configurations."""
    storage = {}

    # Hearts chest
    storage[OBJECT_IDS["hearts_chest"]] = ConverterConfig(
        type_id=OBJECT_IDS["hearts_chest"],
        input_resources={},
        output_resources={},
        max_output=-1,  # Unlimited storage
        max_conversions=-1,
        conversion_ticks=1,
        cooldown=0,
        color=6
    )

    # Resource storage depots
    for resource, resource_id in RESOURCE_IDS.items():
        if resource == "heart":
            continue  # Hearts use the chest

        depot_id = OBJECT_IDS[f"{resource}_depot"]
        storage[depot_id] = ConverterConfig(
            type_id=depot_id,
            input_resources={},
            output_resources={},
            max_output=-1,
            max_conversions=-1,
            conversion_ticks=1,
            cooldown=0,
            color=7
        )

    return storage


def create_cogs_vs_clips_map() -> str:
    """Generate ASCII map for Cogs vs Clips with 20 CoGS in center."""
    # 40x40 map with central nano-assembler and surrounding CoGS
    map_lines = []

    # Initialize empty map
    for row in range(40):
        line = ['.'] * 40
        map_lines.append(line)

    # Place nano-assembler at center (20, 20)
    center_r, center_c = 20, 20
    map_lines[center_r][center_c] = 'N'  # Nano-assembler

    # Place 20 CoGS around the center in a 5x5 pattern (excluding center)
    cog_positions = [
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        (0, -2), (0, -1), (0, 1), (0, 2),  # Skip (0,0) - nano-assembler
        (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
        (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
    ]

    # Take first 20 positions for CoGS
    for i, (dr, dc) in enumerate(cog_positions[:20]):
        r, c = center_r + dr, center_c + dc
        if 0 <= r < 40 and 0 <= c < 40:
            map_lines[r][c] = 'A'  # Agent/CoGS

    # Place resource patches around the map
    import random
    random.seed(42)  # Deterministic placement

    # Rich resource patches (closer to center)
    for _ in range(8):
        while True:
            r, c = random.randint(8, 32), random.randint(8, 32)
            if map_lines[r][c] == '.':
                map_lines[r][c] = random.choice(['R', 'H', 'M'])  # Rare earth, Helium, Mercury
                break

    # Solar stations (batteries)
    for _ in range(6):
        while True:
            r, c = random.randint(5, 35), random.randint(5, 35)
            if map_lines[r][c] == '.':
                map_lines[r][c] = 'S'  # Solar
                break

    # Depleted patches (scattered)
    for _ in range(12):
        while True:
            r, c = random.randint(2, 38), random.randint(2, 38)
            if map_lines[r][c] == '.':
                map_lines[r][c] = random.choice(['r', 'h', 'm'])  # lowercase = depleted
                break

    # Storage facilities at corners and edges
    storage_positions = [
        (2, 2), (2, 37), (37, 2), (37, 37),  # Corners
        (2, 20), (37, 20), (20, 2), (20, 37)  # Edges
    ]
    storage_symbols = ['C', 'E', 'L', 'U', 'B', 'F', 'G', 'I']  # Different storage types

    for (r, c), symbol in zip(storage_positions, storage_symbols):
        if map_lines[r][c] == '.':
            map_lines[r][c] = symbol

    # Convert to string
    return '\n'.join(''.join(line) for line in map_lines)


def cogs_vs_clips_env(config: CogsVsClipsConfig = None) -> GameConfig:
    """
    Create a complete Cogs vs Clips game environment configuration.

    Args:
        config: Game configuration parameters. Uses defaults if None.

    Returns:
        GameConfig: Complete mettagrid configuration ready for gameplay.
    """
    if config is None:
        config = CogsVsClipsConfig()

    # Create all object configurations
    objects = {}

    # Add nano-assembler
    objects["nano_assembler"] = create_nano_assembler_config()

    # Add resource extractors
    extractors = create_resource_extractors()
    for obj_id, extractor_config in extractors.items():
        objects[str(obj_id)] = extractor_config

    # Add storage facilities
    storage = create_storage_facilities()
    for obj_id, storage_config in storage.items():
        objects[str(obj_id)] = storage_config

    # Create the main game configuration
    game_config = GameConfig(
        resource_names=list(RESOURCE_IDS.keys()),
        num_agents=config.max_agents,
        max_steps=config.max_steps,
        obs_width=7,
        obs_height=7,
        objects=objects,
    )

    return game_config


# Symbol mapping for ASCII map builder
MAP_SYMBOLS = {
    'N': OBJECT_IDS["nano_assembler"],
    'A': "agent",  # Special case - agents are handled separately
    'R': OBJECT_IDS["rare_earth_rich"],
    'H': OBJECT_IDS["trapped_helium_rich"],
    'M': OBJECT_IDS["mercury_rich"],
    'S': OBJECT_IDS["battery_solar"],
    'r': OBJECT_IDS["rare_earth_depleted"],
    'h': OBJECT_IDS["trapped_helium_depleted"],
    'm': OBJECT_IDS["mercury_depleted"],
    'C': OBJECT_IDS["hearts_chest"],
    'E': OBJECT_IDS["rare_earth_depot"],
    'L': OBJECT_IDS["trapped_helium_depot"],
    'U': OBJECT_IDS["mercury_depot"],
    'B': OBJECT_IDS["battery_depot"],
    '.': None,  # Empty space
}


def get_symbol_mapping():
    """Get the symbol to object ID mapping for the map builder."""
    return MAP_SYMBOLS


if __name__ == "__main__":
    # Example usage
    config = CogsVsClipsConfig(max_agents=20, grid_size=(40, 40))
    env_config = cogs_vs_clips_env(config)
    print("Cogs vs Clips environment created successfully!")
    print(f"Grid size: {env_config.grid_size}")
    print(f"Max agents: {env_config.max_agents}")
    print(f"Object types: {len(env_config.objects)}")
    print(f"Map preview (first 5 lines):")
    print('\n'.join(env_config.initial_map.split('\n')[:5]))