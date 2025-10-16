from cogames.cogs_vs_clips import protocols
from cogames.cogs_vs_clips.stations import (
    assembler,
)
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    MettaGridConfig,
    RecipeConfig,
)


def replace_assembler_recipe_simple(cfg: MettaGridConfig) -> MettaGridConfig:
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
    ]
    return cfg


def replace_assembler_recipe_complex(cfg: MettaGridConfig) -> MettaGridConfig:
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
    ]
    return cfg


def add_easy_heart_recipe(cfg: MettaGridConfig) -> None:
    """Insert a simple energy-to-heart recipe for the assembler."""

    assembler_cfg: AssemblerConfig | None = cfg.game.objects.get("assembler")  # type: ignore
    if assembler_cfg is None:
        return

    for _, recipe in assembler_cfg.recipes:
        if recipe.output_resources.get("heart") and recipe.input_resources == {"energy": 1}:
            return

    easy_recipe = RecipeConfig(
        input_resources={"energy": 1},
        output_resources={"heart": 1},
        cooldown=1,
    )
    assembler_cfg.recipes += protocols.protocol(easy_recipe, num_agents=1)
    assembler_cfg.fully_overlapping_recipes_allowed = True


def add_shaped_rewards(cfg: MettaGridConfig) -> None:
    """Augment agent rewards with additional heart-centric shaped rewards."""

    agent_cfg = cfg.game.agent
    stats = dict(agent_cfg.rewards.stats or {})

    stats["heart.gained"] = 5.0
    stats["heart.put"] = 7.5
    stats["chest.heart.amount"] = 2.5

    shaped_reward = 0.25
    stats.update(
        {
            "carbon.gained": shaped_reward,
            "oxygen.gained": shaped_reward,
            "germanium.gained": shaped_reward,
            "silicon.gained": shaped_reward,
            "energy.gained": shaped_reward / 5,
        }
    )

    agent_cfg.rewards.stats = stats


def extend_max_steps(cfg: MettaGridConfig, multiplier: float = 20.0) -> None:
    current = cfg.game.max_steps
    if current is None:
        return
    cfg.game.max_steps = int(current * multiplier)


_MODIFICATIONS = {
    "easy": add_easy_heart_recipe,
    "shaped": add_shaped_rewards,
    "extend_max_steps": extend_max_steps,
    "replace_assembler_recipe_simple": replace_assembler_recipe_simple,
    "replace_assembler_recipe_complex": replace_assembler_recipe_complex,
}


def apply_modifications(cfg: MettaGridConfig, modifications: dict[str, bool]) -> None:
    for modification, value in modifications.items():
        if value is True and modification in _MODIFICATIONS:
            _MODIFICATIONS[modification](cfg)
