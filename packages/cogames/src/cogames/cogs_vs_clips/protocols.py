from mettagrid.config.mettagrid_config import RecipeConfig


def standard_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 5, "silicon": 50, "energy": 20},
        output_resources={"heart": 1},
        cooldown=1,
    )


# We might want this to just make more hearts, but agent inventory is limited, so cheaper is better.
def low_germanium_heart_recipe() -> RecipeConfig:
    return RecipeConfig(
        input_resources={"carbon": 20, "oxygen": 20, "germanium": 3, "silicon": 50, "energy": 20},
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
def standard_germanium_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"germanium": 1},
    )


def low_germanium_recipe() -> RecipeConfig:
    return RecipeConfig(
        output_resources={"germanium": 1},
    )
