from typing import Optional

from pydantic import BaseModel

from cogames.cogs_vs_clips.stations import (
    assembler,
    carbon_ex_dep,
    carbon_extractor,
    charger,
    chest,
    chest_carbon,
    chest_germanium,
    chest_oxygen,
    chest_silicon,
    clipped_carbon_extractor,
    clipped_germanium_extractor,
    clipped_oxygen_extractor,
    clipped_silicon_extractor,
    germanium_ex_dep,
    germanium_extractor,
    oxygen_ex_dep,
    oxygen_extractor,
    resources,
    silicon_ex_dep,
    silicon_extractor,
)
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    ClipperConfig,
    GameConfig,
    RecipeConfig,
    WallConfig,
)


class Mission(BaseModel):
    name: str
    description: str
    game: GameConfig


def _default_mission(num_agents: Optional[int] = None) -> Mission:
    game = GameConfig(
        resource_names=resources,
        num_agents=num_agents or 4,
        actions=ActionsConfig(
            move=ActionConfig(consumed_resources={"energy": 2}),
            noop=ActionConfig(),
            change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
        ),
        objects={
            "wall": WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛"),
            "charger": charger(),
            "carbon_extractor": carbon_extractor(),
            "oxygen_extractor": oxygen_extractor(),
            "germanium_extractor": germanium_extractor(),
            "silicon_extractor": silicon_extractor(),
            # depleted variants
            "silicon_ex_dep": silicon_ex_dep(),
            "oxygen_ex_dep": oxygen_ex_dep(),
            "carbon_ex_dep": carbon_ex_dep(),
            "germanium_ex_dep": germanium_ex_dep(),
            "clipped_carbon_extractor": clipped_carbon_extractor(),
            "clipped_oxygen_extractor": clipped_oxygen_extractor(),
            "clipped_germanium_extractor": clipped_germanium_extractor(),
            "clipped_silicon_extractor": clipped_silicon_extractor(),
            "chest": chest(),
            "chest_carbon": chest_carbon(),
            "chest_oxygen": chest_oxygen(),
            "chest_germanium": chest_germanium(),
            "chest_silicon": chest_silicon(),
            "assembler": assembler(),
        },
        agent=AgentConfig(
            resource_limits={
                "heart": 1,
                "energy": 100,
                ("carbon", "oxygen", "germanium", "silicon"): 100,
                ("scrambler", "modulator", "decoder", "resonator"): 5,
            },
            rewards=AgentRewards(
                stats={"chest.heart.amount": 1},
                # inventory={
                #     "heart": 1,
                # },
            ),
            initial_inventory={
                "energy": 100,
            },
            shareable_resources=["energy"],
            inventory_regen_amounts={"energy": 1},
        ),
        inventory_regen_interval=1,
        # Enable clipper system to allow start_clipped assemblers to work
        clipper=ClipperConfig(
            unclipping_recipes=[
                RecipeConfig(
                    input_resources={"decoder": 1},
                    cooldown=1,
                ),
                RecipeConfig(
                    input_resources={"modulator": 1},
                    cooldown=1,
                ),
                RecipeConfig(
                    input_resources={"scrambler": 1},
                    cooldown=1,
                ),
                RecipeConfig(
                    input_resources={"resonator": 1},
                    cooldown=1,
                ),
            ],
            length_scale=10.0,
            cutoff_distance=0.0,
            clip_rate=0.0,  # Don't clip during gameplay, only use start_clipped
        ),
    )

    return Mission(name="default", description="Default mission", game=game)


def energy_intensive(num_agents: Optional[int] = None) -> Mission:
    mission = _default_mission(num_agents)
    mission.name = "energy_intensive"
    mission.description = "Energy intensive mission"
    mission.game.actions.move.consumed_resources = {"energy": 5}
    mission.game.agent.resource_limits.update(
        {
            "heart": 1,
            ("carbon", "oxygen", "germanium", "silicon"): 2,
            ("scrambler", "modulator", "decoder", "resonator"): 5,
        }
    )
    mission.game.agent.inventory_regen_amounts = {"energy": 3}
    return mission


def get_all_missions(num_agents: Optional[int] = None) -> dict[str, Mission]:
    return {
        "default": _default_mission(num_agents),
        "energy_intensive": energy_intensive(num_agents),
    }
