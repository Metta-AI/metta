"""Cog Arena recipe - STABLE

A complete arena configuration showcasing:
- 6 vibes: default, weapon, shield, battery, gear, heart
- Complex resource/crafting economy with assemblers
- Damage accumulation system
- Vibe-triggered attack and transfer mechanics

This recipe is automatically validated in CI and release processes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCStationConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.config import DamageConfig, VibeTransfer
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AssemblerConfig,
    AttackActionConfig,
    ChangeVibeActionConfig,
    ChestConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
    TransferActionConfig,
)
from mettagrid.config.vibes import Vibe
from mettagrid.mapgen.mapgen import MapGen

# Resources available in the cog arena
RESOURCES = [
    "energy",
    "carbon",
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "weapon",
    "shield",
    "battery",
    "gear",
    "damage",
]

# Vibes available to agents
VIBES = [
    Vibe("😐", "default"),
    Vibe("⚔️", "weapon"),
    Vibe("🛡️", "shield"),
    Vibe("🔋", "battery"),
    Vibe("⚙️", "gear"),
    Vibe("❤️", "heart"),
]


class CogAssemblerConfig(CvCStationConfig):
    """Assembler configuration for crafting in the cog arena.

    Supports crafting:
    - Hearts from basic resources (heart vibe)
    - Weapons from carbon (weapon vibe)
    - Shields from silicon (shield vibe)
    - Batteries from germanium + silicon (battery vibe)
    - Gear from oxygen + silicon + carbon + germanium (gear vibe)
    """

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="assembler",
            render_symbol="",
            clip_immune=True,
            chest_search_distance=10,
            protocols=[
                ProtocolConfig(
                    vibes=["heart"],
                    input_resources={
                        "carbon": 10,
                        "oxygen": 10,
                        "germanium": 1,
                        "silicon": 50,
                    },
                    output_resources={"heart": 1},
                    sigmoid=3,
                    inflation=0.1,
                ),
                ProtocolConfig(
                    vibes=["weapon"],
                    input_resources={"carbon": 5},
                    output_resources={"weapon": 1},
                    sigmoid=3,
                    inflation=0.1,
                ),
                ProtocolConfig(
                    vibes=["shield"],
                    input_resources={"silicon": 25},
                    output_resources={"shield": 1},
                    sigmoid=3,
                    inflation=0.1,
                ),
                ProtocolConfig(
                    vibes=["battery"],
                    input_resources={"germanium": 5, "silicon": 5},
                    output_resources={"battery": 1},
                    sigmoid=3,
                    inflation=0.1,
                ),
                ProtocolConfig(
                    vibes=["gear"],
                    input_resources={"oxygen": 10, "silicon": 50, "carbon": 10, "germanium": 5},
                    output_resources={"gear": 1},
                    sigmoid=5,
                    inflation=0.1,
                ),
            ],
            agent_cooldown=10,
        )


def make_env(num_agents: int = 10) -> MettaGridConfig:
    """Create a cog arena environment configuration.

    Args:
        num_agents: Number of agents in the environment.

    Returns:
        A fully configured MettaGridConfig for the cog arena.
    """
    map_builder = MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
        ),
    )
    game = GameConfig(
        map_builder=map_builder,
        max_steps=1000,
        num_agents=num_agents,
        resource_names=RESOURCES,
        vibe_names=[vibe.name for vibe in VIBES],
        global_obs=GlobalObsConfig(),
        actions=ActionsConfig(
            move=MoveActionConfig(
                consumed_resources={"energy": 3},
            ),
            noop=NoopActionConfig(),
            change_vibe=ChangeVibeActionConfig(number_of_vibes=len(VIBES), vibes=VIBES),
            attack=AttackActionConfig(
                consumed_resources={"energy": 7},
                defense_resources={"energy": 0},
                weapon_resources={"weapon": 10},
                armor_resources={"shield": 15},
                loot=["heart"],
                enabled=False,
                vibes=["weapon"],  # Attack triggered when agent has weapon vibe
            ),
            transfer=TransferActionConfig(
                vibe_transfers=[
                    VibeTransfer(vibe="battery", target={"energy": 50}, actor={"energy": -50}),
                    VibeTransfer(vibe="heart", target={"heart": 1}, actor={"heart": -1}),
                    VibeTransfer(vibe="gear", target={"damage": -100}, actor={"energy": -10}),
                ],
                vibes=["battery", "heart", "gear"],  # Transfer triggered for these vibes
            ),
        ),
        agent=AgentConfig(
            resource_limits={
                "heart": ResourceLimitsConfig(limit=100, resources=["heart"]),
                "energy": ResourceLimitsConfig(limit=0, resources=["energy"], modifiers={"battery": 25}),
                "cargo": ResourceLimitsConfig(limit=255, resources=["carbon", "oxygen", "germanium", "silicon"]),
                "gear": ResourceLimitsConfig(limit=0, resources=["weapon", "shield", "battery"], modifiers={"gear": 1}),
            },
            rewards=AgentRewards(
                inventory={
                    "heart": 1,
                    "carbon": 0.001,
                    "oxygen": 0.001,
                    "germanium": 0.001,
                    "silicon": 0.001,
                    "weapon": 0.01,
                    "shield": 0.01,
                    "battery": 0.01,
                    "gear": 0.1,
                },
            ),
            initial_inventory={
                "gear": 10,
                "battery": 4,
                "energy": 100,
                "weapon": 1,
                "shield": 0,
                "heart": 1,
                "silicon": 50,
                "oxygen": 50,
                "carbon": 50,
                "germanium": 50,
            },
            # Order matters: gear modifies battery/weapon/shield limit, battery modifies energy limit
            inventory_order=["gear", "battery", "weapon", "shield"],
            damage=DamageConfig(
                threshold={"damage": 200},
                resources={"battery": 1, "weapon": 0, "shield": 0},  # resource -> minimum (won't go below)
            ),
            inventory_regen_amounts={
                "default": {"energy": 1, "damage": 1},
            },
            freeze_duration=5,
        ),
        inventory_regen_interval=1,
        objects={
            "wall": CvCWallConfig().station_cfg(),
            "assembler": CogAssemblerConfig().station_cfg(),
            "charger": ChargerConfig().station_cfg(),
            "chest": ChestConfig(
                vibe_transfers={
                    "default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255},
                }
            ),
            "carbon_extractor": CarbonExtractorConfig().station_cfg(),
            "oxygen_extractor": OxygenExtractorConfig(efficiency=200).station_cfg(),
            "germanium_extractor": GermaniumExtractorConfig().station_cfg(),
            "silicon_extractor": SiliconExtractorConfig().station_cfg(),
        },
    )

    return MettaGridConfig(game=game)


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create a curriculum for training on the cog arena.

    Args:
        arena_env: Optional base environment configuration.
        algorithm_config: Optional curriculum algorithm configuration.

    Returns:
        A curriculum configuration with task buckets for progressive training.
    """
    arena_env = arena_env or make_env()

    arena_tasks = cc.bucketed(arena_env)

    # Curriculum buckets for progressive difficulty
    arena_tasks.add_bucket("game.max_steps", [1000, 5000, 10000])
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.energy", [7, 8, 9, 10])
    arena_tasks.add_bucket("game.actions.attack.weapon_resources.weapon", [9, 10, 11])
    arena_tasks.add_bucket("game.actions.attack.armor_resources.shield", [12, 13, 14])
    arena_tasks.add_bucket("game.agent.initial_inventory.weapon", [0, 1, 2, 3])
    arena_tasks.add_bucket("game.agent.initial_inventory.shield", [0, 1, 2, 3])
    arena_tasks.add_bucket("game.agent.initial_inventory.battery", [3, 4, 5, 6])
    arena_tasks.add_bucket("game.agent.inventory_regen_amounts.default.damage", [1, 2, 3])
    arena_tasks.add_bucket("game.agent.inventory_regen_amounts.default.energy", [1, 2, 3])

    if algorithm_config is None:
        algorithm_config = DiscreteRandomConfig()

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation simulations for the cog arena.

    Args:
        env: Optional environment configuration.

    Returns:
        List of simulation configurations for evaluation.
    """
    basic_env = env or make_env()

    return [
        SimulationConfig(suite="cog_arena", name="basic", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    """Create a training tool for the cog arena.

    Args:
        curriculum: Optional curriculum configuration.

    Returns:
        A TrainTool configured for the cog arena.
    """
    resolved_curriculum = curriculum or make_curriculum()
    trainer_cfg = TrainerConfig()
    evaluator_cfg = EvaluatorConfig(simulations=simulations())

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Create an evaluation tool for the cog arena.

    Args:
        policy_uris: Policy URIs to evaluate.

    Returns:
        An EvaluateTool configured for the cog arena.
    """
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris or [],
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy in the cog arena.

    Args:
        policy_uri: Optional policy URI to play with.

    Returns:
        A PlayTool configured for the cog arena.
    """
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy in the cog arena.

    Args:
        policy_uri: Optional policy URI to replay.

    Returns:
        A ReplayTool configured for the cog arena.
    """
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
