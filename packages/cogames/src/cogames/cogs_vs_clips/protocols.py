from mettagrid.config.mettagrid_config import Position, RecipeConfig


def protocol(
    recipe: RecipeConfig,
    num_agents: int | None = None,
    min_agents: int | None = None,
    max_agents: int | None = None,
    fixed_positions: list[Position] | None = None,
) -> list[tuple[list[Position], RecipeConfig]]:
    if num_agents is not None:
        assert min_agents is None
        assert max_agents is None
        min_agents = num_agents
        max_agents = num_agents
    if min_agents is None:
        min_agents = 1
    if max_agents is None:
        max_agents = 8
    result = []
    if fixed_positions is None:
        fixed_positions = []
    for num_agents in range(min_agents, max_agents + 1):
        num_any = num_agents - len(fixed_positions)
        if num_any < 0:
            continue
        result.append((fixed_positions + ["Any"] * num_any, recipe))
    return result


def one_agent_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 5, "silicon": 50, "energy": 20},
        output_resources={"heart": 1},
        cooldown=1,
    )


def two_agent_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 4, "silicon": 50, "energy": 20},
        output_resources={"heart": 1},
        cooldown=1,
    )


def three_agent_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 3, "silicon": 50, "energy": 20},
        output_resources={"heart": 1},
        cooldown=1,
    )


def four_agent_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 2, "silicon": 50, "energy": 20},
        output_resources={"heart": 1},
        cooldown=1,
    )


def standard_charging_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"energy": 50},
        cooldown=10,
    )


# Carbon takes time.
def standard_carbon_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"carbon": 4},
    )


def low_carbon_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"carbon": 1},
    )


# Oxygen refreshes somewhat slowly and takes space.
def standard_oxygen_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"oxygen": 100},
        cooldown=200,
    )


def low_oxygen_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"oxygen": 10},
        cooldown=40,
    )


# Silicon is plentiful but requires energy / work and need a lot.
def standard_silicon_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"energy": 25},
        output_resources={"silicon": 25},
    )


def low_silicon_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"energy": 25},
        output_resources={"silicon": 10},
    )


# Germanium is rare and exhausts / regenerates slowly.
# Outputs are always low, and the different between high and low is the number of uses.
def germanium_recipe(num_agents: int) -> RecipeConfig:
    amount = 1 + 1 * (min(num_agents, 4))
    return RecipeConfig(
        output_resources={"germanium": amount},
    )


# Equipment recipes - craft specialized gear from base resources


def decoder_recipe() -> RecipeConfig:
    """Decoder - crafted from germanium."""
    return RecipeConfig(
        input_resources={"germanium": 5},
        output_resources={"decoder": 1},
        cooldown=1,
    )


def modulator_recipe() -> RecipeConfig:
    """Modulator - crafted from carbon."""
    return RecipeConfig(
        input_resources={"carbon": 50},
        output_resources={"modulator": 1},
        cooldown=1,
    )


def resonator_recipe() -> RecipeConfig:
    """Resonator - crafted from silicon."""
    return RecipeConfig(
        input_resources={"silicon": 100},
        output_resources={"resonator": 1},
        cooldown=1,
    )


def scrambler_recipe() -> RecipeConfig:
    """Scrambler - crafted from oxygen."""
    return RecipeConfig(
        input_resources={"oxygen": 50},
        output_resources={"scrambler": 1},
        cooldown=1,
    )
