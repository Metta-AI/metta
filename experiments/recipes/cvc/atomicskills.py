"""Atomic Skills Curriculum for CoGs vs Clips.

This curriculum breaks down CVC tasks into atomic skills that teach agents
fundamental behaviors through progressive difficulty. Skills are organized
into clusters and sampled using learning progress curriculum.

Skill Progression:
1. Basic awareness (heart/chest recognition)
2. Simple interactions (chest deposit, assembler usage)
3. Resource management (collection, conversion)
4. Navigation (pathfinding, home return)
5. Efficiency (timing, opportunity cost)
6. Coordination (synchronization, role specialization)
"""

from __future__ import annotations

from typing import ClassVar, Optional

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.evals import (
    CANONICAL_DIFFICULTY_ORDER,
    DIFFICULTY_LEVELS,
    apply_difficulty,
)
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import EVALS, Site
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import (
    CheckpointerConfig,
    EvaluatorConfig,
    TrainingEnvironmentConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeVibeActionConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder


# ============================================================================
# Atomic Skill Mission Definitions
# ============================================================================


class AtomicSkillMission(Mission):
    """Base class for atomic skill missions with common configurations."""

    # All atomic skill missions use the EVALS site for consistency
    site: Site = EVALS

    # Small map for focused learning
    default_map_width: int = 15
    default_map_height: int = 15

    # Simplified agent config for atomic skills
    cargo_capacity: int = 255
    energy_capacity: int = 100
    energy_regen_amount: int = 1
    move_energy_cost: int = 2
    heart_capacity: int = 10
    clip_rate: float = 0.0  # No clipping for atomic skills
    enable_vibe_change: bool = True

    # CRITICAL: All missions in a curriculum must use the SAME resource names
    # and object types for the simulator's config invariants check
    STANDARD_RESOURCE_NAMES: ClassVar[list[str]] = [
        "carbon",
        "oxygen",
        "germanium",
        "silicon",
        "heart",
        "energy",
        # Gear resources for assembler protocols
        "decoder",
        "modulator",
        "scrambler",
        "resonator",
    ]

    def configure(self) -> None:
        """Initialize all station configs so they're available for object dict."""
        super().configure()
        # Ensure all station configs exist even if not explicitly set
        if not hasattr(self, "carbon_extractor"):
            self.carbon_extractor = CarbonExtractorConfig(efficiency=100, max_uses=1000)
        if not hasattr(self, "assembler"):
            self.assembler = CvCAssemblerConfig(heart_cost=5)
        if not hasattr(self, "chest"):
            self.chest = CvCChestConfig(initial_inventory={})

    def get_standard_objects(self) -> dict:
        """Get standard objects dict that all missions must include.

        Even if a mission doesn't use certain objects, they must all be present
        for the simulator's config invariants check.
        """
        return {
            "wall": self.wall.station_cfg(),
            "chest": self.chest.station_cfg(),
            "assembler": self.assembler.station_cfg(),
            "carbon_extractor": self.carbon_extractor.station_cfg(),
        }

    def get_simple_map(self, width: int, height: int, num_cogs: int):  # type: ignore[no-untyped-def]
        """Create a simple procedural map for atomic skills."""
        return RandomMapBuilder.Config(
            agents=num_cogs,
            width=width,
            height=height,
            border_width=1,
        )


class HeartAwarenessMission(AtomicSkillMission):
    """Learn to recognize heart in inventory.

    Agent spawns with heart, no chest. Reward only for holding heart.
    """

    name: str = "heart_awareness"
    description: str = "Recognize and value the heart in your inventory"

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        # Empty map, just spawn with heart
        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            # vibe_names auto-populated from VIBES[:8]
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                # All missions need same action space for curriculum learning
                change_vibe=ChangeVibeActionConfig(number_of_vibes=8),
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight},  # Reward for holding heart
                ),
                initial_inventory={
                    "heart": 1,  # Start with one heart
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class ChestDiscoveryMission(AtomicSkillMission):
    """Learn to locate and interact with chest.

    Chest exists but starts empty. Small reward for opening chest.
    """

    name: str = "chest_discovery"
    description: str = "Find and interact with the chest"

    # Override chest to have only heart vibe transfers (no carbon/oxygen/etc)
    def configure(self):
        # Chest starts empty
        self.chest = CvCChestConfig(initial_inventory={})

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        # Simple map with one chest
        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            # vibe_names auto-populated from VIBES[:7]
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                # All missions need same action space (8) for curriculum learning
                change_vibe=ChangeVibeActionConfig(number_of_vibes=8),
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight * 0.5},
                    stats={
                        "chest.heart.deposited": reward_weight
                    },  # Reward for depositing
                ),
                initial_inventory={
                    "heart": 1,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class Chest101Mission(AtomicSkillMission):
    """Learn to deposit heart in chest for growth.

    Agent starts with heart. Chest gains interest over time.
    Reward = inventory hearts + chest hearts.
    """

    name: str = "chest101"
    description: str = "Deposit heart in chest to gain interest over time"

    # Override chest to have only heart vibe transfers (no carbon/oxygen/etc)
    def configure(self):
        # Chest starts empty, will accumulate hearts
        self.chest = CvCChestConfig(initial_inventory={})

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            # vibe_names auto-populated from VIBES[:7]
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                # All missions need same action space (8) for curriculum learning
                change_vibe=ChangeVibeActionConfig(number_of_vibes=8),
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight},
                    stats={
                        "chest.heart.amount": reward_weight
                    },  # Shared reward from chest
                ),
                initial_inventory={
                    "heart": 1,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class ResourceAwarenessMission(AtomicSkillMission):
    """Learn that resources exist in inventory.

    Start with 1-2 resources, learn they have value.
    """

    name: str = "resource_awareness"
    description: str = "Recognize resources in your inventory"

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            # vibe_names auto-populated from VIBES[:8]
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                # All missions need same action space for curriculum learning
                change_vibe=ChangeVibeActionConfig(number_of_vibes=8),
            ),
            agent=AgentConfig(
                resource_limits={
                    "carbon": self.cargo_capacity,
                    "oxygen": self.cargo_capacity,
                    "energy": self.energy_capacity,
                },
                rewards=AgentRewards(
                    inventory={
                        "carbon": reward_weight * 0.1,
                        "oxygen": reward_weight * 0.1,
                    },
                ),
                initial_inventory={
                    "carbon": 2,
                    "oxygen": 2,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class ResourceToHeartMission(AtomicSkillMission):
    """Learn to convert resources into heart at assembler.

    Start with all 4 resources, find assembler, make heart, deposit in chest.
    """

    name: str = "resource_to_heart"
    description: str = "Convert resources into a heart at the assembler"

    def configure(self):
        # Simple assembler with reduced costs for learning
        self.assembler = CvCAssemblerConfig(heart_cost=5)
        self.chest = CvCChestConfig(initial_inventory={})

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    number_of_vibes=8
                ),  # Need 8 to include gear (index 7)
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight * 0.5},
                    stats={"chest.heart.amount": reward_weight},
                ),
                initial_inventory={
                    "carbon": 20,
                    "oxygen": 20,
                    "germanium": 5,
                    "silicon": 30,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class HarvestOneResourceMission(AtomicSkillMission):
    """Learn to harvest one resource from chest.

    Have 3 of 4 resources, need to get one from chest to make heart.
    """

    name: str = "harvest_one_resource"
    description: str = "Extract one resource from chest to complete recipe"

    def configure(self):
        # Chest with one resource type
        self.chest = CvCChestConfig(initial_inventory={"carbon": 20})
        self.assembler = CvCAssemblerConfig(heart_cost=5)

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    number_of_vibes=8
                ),  # Need 8 to include gear (index 7)
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight},
                ),
                initial_inventory={
                    "oxygen": 20,
                    "germanium": 5,
                    "silicon": 30,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class ExtractorDiscoveryMission(AtomicSkillMission):
    """Learn to find and use extractors.

    Single extractor, single resource type, close proximity.
    """

    name: str = "extractor_discovery"
    description: str = "Find and use an extractor to get resources"

    def configure(self):
        self.carbon_extractor = CarbonExtractorConfig(efficiency=100, max_uses=1000)
        self.assembler = CvCAssemblerConfig(heart_cost=5)
        self.chest = CvCChestConfig(initial_inventory={})

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=self.default_map_width,
            height=self.default_map_height,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    number_of_vibes=8
                ),  # Need 8 to include gear (index 7)
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight * 0.5},
                    stats={"chest.heart.amount": reward_weight},
                ),
                initial_inventory={
                    "oxygen": 20,
                    "germanium": 5,
                    "silicon": 30,
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


class ExtractorUsageMission(AtomicSkillMission):
    """Learn to get 3 resources from chests, 1 from extractor.

    Combines chest extraction with extractor usage.
    """

    name: str = "extractor_usage"
    description: str = "Get resources from both chests and extractors"

    def configure(self):
        self.carbon_extractor = CarbonExtractorConfig(efficiency=100, max_uses=1000)
        self.chest = CvCChestConfig(
            initial_inventory={
                "oxygen": 20,
                "germanium": 5,
                "silicon": 30,
            }
        )
        self.assembler = CvCAssemblerConfig(heart_cost=5)

    def make_env(self) -> MettaGridConfig:
        if self.num_cogs is None:
            raise ValueError("num_cogs must be set")

        map_builder = RandomMapBuilder.Config(
            agents=self.num_cogs,
            width=20,
            height=20,
            border_width=1,
        )

        reward_weight = 1.0 / self.num_cogs

        game = GameConfig(
            map_builder=map_builder,
            num_agents=self.num_cogs,
            resource_names=self.STANDARD_RESOURCE_NAMES,
            actions=ActionsConfig(
                move=MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    number_of_vibes=8
                ),  # Need 8 to include gear (index 7)
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                },
                rewards=AgentRewards(
                    inventory={"heart": reward_weight * 0.5},
                    stats={"chest.heart.amount": reward_weight},
                ),
                initial_inventory={
                    "energy": self.energy_capacity,
                },
            ),
            objects=self.get_standard_objects(),
        )

        return MettaGridConfig(game=game, label=self.name)


# List of all atomic skill missions
ATOMIC_SKILL_MISSIONS = [
    HeartAwarenessMission,
    ChestDiscoveryMission,
    Chest101Mission,
    ResourceAwarenessMission,
    ResourceToHeartMission,
    HarvestOneResourceMission,
    ExtractorDiscoveryMission,
    ExtractorUsageMission,
]


# ============================================================================
# Evaluation Suite Creation
# ============================================================================


def make_standard_cvc_eval_suite(
    num_cogs: int = 4,
    subset: Optional[list[str]] = None,
    difficulties: Optional[list[str]] = None,
) -> list[SimulationConfig]:
    """Create standard CVC evaluation suite (same as scripted agent).

    This uses the same 13 canonical CVC missions used to evaluate the
    scripted baseline agents, enabling direct comparison.

    Args:
        num_cogs: Number of agents per mission (1, 2, 4, or 8)
        subset: Optional list of mission names to include (defaults to all)
        difficulties: List of difficulty names to test (defaults to standard only).
                     For full sweep, pass CANONICAL_DIFFICULTY_ORDER.

    Returns:
        List of SimulationConfig objects for evaluation
    """
    # Default to standard difficulty only for faster evaluation
    if difficulties is None:
        difficulties = ["standard"]

    # Filter missions if subset specified
    if subset:
        missions = [m for m in EVAL_MISSIONS if m().name in subset]
    else:
        missions = EVAL_MISSIONS

    simulations = []
    for difficulty_name in difficulties:
        if difficulty_name not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Unknown difficulty: {difficulty_name}. Valid: {list(DIFFICULTY_LEVELS.keys())}"
            )

        difficulty_level = DIFFICULTY_LEVELS[difficulty_name]

        for mission_cls in missions:
            mission_template = mission_cls()  # type: ignore[call-arg]

            # Skip missions that don't make sense for single agent
            if num_cogs == 1 and mission_template.name in [
                "go_together",
                "single_use_swarm",
            ]:
                continue

            # Get default map for this mission
            map_builder = get_map(mission_template.map_name)

            # Instantiate mission with specified agent count
            instantiated = mission_template.instantiate(
                map_builder=map_builder,
                num_cogs=num_cogs,
                variant=None,
            )

            # Apply difficulty modifiers
            apply_difficulty(instantiated, difficulty_level)

            # Create env config
            env_cfg = instantiated.make_env()

            # Create simulation with difficulty in name
            sim = SimulationConfig(
                suite="cogs_vs_clips_eval",
                name=f"{mission_template.name}_{num_cogs}cogs_{difficulty_name}",
                env=env_cfg,
            )
            simulations.append(sim)

    return simulations


# ============================================================================
# Curriculum Creation
# ============================================================================


def make_atomic_skill_env(
    mission_cls: type[AtomicSkillMission],
    num_cogs: int = 1,
    map_width: int = 15,
    map_height: int = 15,
) -> MettaGridConfig:
    """Create a single atomic skill training environment.

    Args:
        mission_cls: Mission class to instantiate
        num_cogs: Number of agents
        map_width: Width of the map
        map_height: Height of the map

    Returns:
        MettaGridConfig ready for training
    """
    mission = mission_cls()  # type: ignore[call-arg]
    mission.configure()

    # Create simple empty room map
    map_builder = RandomMapBuilder.Config(
        agents=num_cogs,
        width=map_width,
        height=map_height,
        border_width=1,  # Add border walls
    )

    # Instantiate mission
    mission.map = map_builder
    mission.num_cogs = num_cogs

    return mission.make_env()


def make_curriculum(
    num_cogs: int = 1,
    skill_missions: Optional[list[type[AtomicSkillMission]]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create atomic skills curriculum.

    This creates a curriculum that teaches fundamental CVC skills through
    progressive difficulty. Each skill cluster can be sampled based on
    learning progress.

    Args:
        num_cogs: Number of agents (typically 1-4 for atomic skills)
        skill_missions: List of mission classes to include (defaults to all)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        algorithm_config: Curriculum algorithm config (defaults to Learning Progress)

    Returns:
        CurriculumConfig for atomic skills training
    """
    if skill_missions is None:
        skill_missions = ATOMIC_SKILL_MISSIONS

    all_skill_tasks = []

    for mission_cls in skill_missions:
        # Create base env for this skill
        base_env = make_atomic_skill_env(mission_cls=mission_cls, num_cogs=num_cogs)

        # Create bucketed tasks for this skill
        skill_tasks = cc.bucketed(base_env)

        # Add buckets for progressive difficulty

        # Map size variation (spatial complexity)
        skill_tasks.add_bucket("game.map_builder.width", [10, 15, 20, 30])
        skill_tasks.add_bucket("game.map_builder.height", [10, 15, 20, 30])

        # Episode length variation
        skill_tasks.add_bucket("game.max_steps", [500, 750, 1000, 1500])

        # Agent count variation (for coordination)
        if num_cogs > 1:
            skill_tasks.add_bucket("game.num_agents", [1, 2, num_cogs])

        # Energy constraints (efficiency pressure)
        skill_tasks.add_bucket("game.agent.resource_limits.energy", [50, 100, 200])
        skill_tasks.add_bucket("game.actions.move.consumed_resources.energy", [1, 2, 4])

        # Reward shaping
        if "heart" in base_env.game.agent.rewards.inventory:
            skill_tasks.add_bucket(
                "game.agent.rewards.inventory.heart", [0.5, 1.0, 2.0]
            )

        # Initial inventory variation (for resource-based skills)
        resource_names = ["carbon", "oxygen", "germanium", "silicon"]
        for resource in resource_names:
            if resource in base_env.game.resource_names:
                skill_tasks.add_bucket(
                    f"game.agent.initial_inventory.{resource}", [0, 5, 10, 20]
                )

        all_skill_tasks.append(skill_tasks)

    # Merge all skill task sets
    merged_tasks = cc.merge(all_skill_tasks)

    # Configure learning progress algorithm
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.15,  # Higher exploration for diverse skills
            max_memory_tasks=3000,  # Many atomic skills
            max_slice_axes=6,  # Multiple dimensions of variation
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return merged_tasks.to_curriculum(
        num_active_tasks=2000,  # Large pool for diverse atomic skills
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 1,
    curriculum: Optional[CurriculumConfig] = None,
    skill_missions: Optional[list[type[AtomicSkillMission]]] = None,
    enable_detailed_slice_logging: bool = False,
    use_standard_cvc_evals: bool = True,
    eval_difficulties: Optional[list[str]] = None,
) -> TrainTool:
    """Create a training tool for atomic skills.

    Args:
        num_cogs: Number of agents (typically 1-4)
        curriculum: Optional custom curriculum (defaults to make_curriculum)
        skill_missions: Skills to include in training curriculum
        enable_detailed_slice_logging: Enable detailed logging
        use_standard_cvc_evals: If True, use standard CVC eval missions (same as scripted agent).
                                If False, use atomic skill missions for eval.
        eval_difficulties: List of difficulties to evaluate on (e.g., ["standard", "hard"]).
                          For full sweep, pass CANONICAL_DIFFICULTY_ORDER.
                          Defaults to ["standard"] for faster evals.

    Returns:
        TrainTool configured for atomic skills training
    """
    # Create or use provided curriculum
    resolved_curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        skill_missions=skill_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    # Configure trainer
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    # Create evaluation suite
    if use_standard_cvc_evals:
        # Use the same evals as scripted baseline agents for direct comparison
        eval_suite = make_standard_cvc_eval_suite(
            num_cogs=num_cogs,
            difficulties=eval_difficulties,
        )
    else:
        # Use atomic skill missions for eval
        eval_suite = []
        for mission_cls in skill_missions or ATOMIC_SKILL_MISSIONS:
            env = make_atomic_skill_env(mission_cls=mission_cls, num_cogs=num_cogs)
            mission_inst = mission_cls()  # type: ignore[call-arg]
            sim = SimulationConfig(
                suite="cvc_atomic_skills",
                name=f"{mission_inst.name}_{num_cogs}cogs",
                env=env,
            )
            eval_suite.append(sim)

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
        epoch_interval=10,  # Evaluate every 10 epochs (same frequency as scripted agent testing)
    )

    checkpointer_cfg = CheckpointerConfig(
        epoch_interval=10,  # Save checkpoints every 10 epochs (same as evaluator)
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
        checkpointer=checkpointer_cfg,
    )


# ============================================================================
# Convenience Training Recipes
# ============================================================================


def train_basic_skills(
    num_cogs: int = 1,
    use_standard_cvc_evals: bool = True,
    eval_difficulties: Optional[list[str]] = None,
) -> TrainTool:
    """Train on basic awareness and interaction skills.

    Args:
        num_cogs: Number of agents
        use_standard_cvc_evals: Use standard CVC evals (True) or atomic skill evals (False)
        eval_difficulties: Difficulties to evaluate on (defaults to ["standard"])
    """
    return train(
        num_cogs=num_cogs,
        skill_missions=[
            HeartAwarenessMission,
            ChestDiscoveryMission,
            Chest101Mission,
        ],
        use_standard_cvc_evals=use_standard_cvc_evals,
        eval_difficulties=eval_difficulties,
    )


def train_resource_skills(
    num_cogs: int = 1,
    use_standard_cvc_evals: bool = True,
    eval_difficulties: Optional[list[str]] = None,
) -> TrainTool:
    """Train on resource management skills.

    Args:
        num_cogs: Number of agents
        use_standard_cvc_evals: Use standard CVC evals (True) or atomic skill evals (False)
        eval_difficulties: Difficulties to evaluate on (defaults to ["standard"])
    """
    return train(
        num_cogs=num_cogs,
        skill_missions=[
            ResourceAwarenessMission,
            ResourceToHeartMission,
            HarvestOneResourceMission,
        ],
        use_standard_cvc_evals=use_standard_cvc_evals,
        eval_difficulties=eval_difficulties,
    )


def train_extraction_skills(
    num_cogs: int = 1,
    use_standard_cvc_evals: bool = True,
    eval_difficulties: Optional[list[str]] = None,
) -> TrainTool:
    """Train on extractor usage skills.

    Args:
        num_cogs: Number of agents
        use_standard_cvc_evals: Use standard CVC evals (True) or atomic skill evals (False)
        eval_difficulties: Difficulties to evaluate on (defaults to ["standard"])
    """
    return train(
        num_cogs=num_cogs,
        skill_missions=[
            ExtractorDiscoveryMission,
            ExtractorUsageMission,
        ],
        use_standard_cvc_evals=use_standard_cvc_evals,
        eval_difficulties=eval_difficulties,
    )


def train_all_atomic_skills(
    num_cogs: int = 1,
    use_standard_cvc_evals: bool = True,
    eval_difficulties: Optional[list[str]] = None,
) -> TrainTool:
    """Train on all atomic skills with standard CVC evaluations.

    Args:
        num_cogs: Number of agents
        use_standard_cvc_evals: Use standard CVC evals (True) or atomic skill evals (False)
        eval_difficulties: Difficulties to evaluate on (defaults to ["standard"]).
                          For full difficulty sweep (like scripted agent), pass CANONICAL_DIFFICULTY_ORDER.
    """
    return train(
        num_cogs=num_cogs,
        skill_missions=ATOMIC_SKILL_MISSIONS,
        use_standard_cvc_evals=use_standard_cvc_evals,
        eval_difficulties=eval_difficulties,
    )


def train_full_difficulty_sweep(num_cogs: int = 4) -> TrainTool:
    """Train on all atomic skills with full difficulty sweep (same as scripted agent eval).

    This creates the exact same evaluation suite as used for scripted baseline agents,
    enabling direct performance comparison across all 13 difficulties.

    Args:
        num_cogs: Number of agents (default 4, the optimal number for scripted agents)
    """
    return train(
        num_cogs=num_cogs,
        skill_missions=ATOMIC_SKILL_MISSIONS,
        use_standard_cvc_evals=True,
        eval_difficulties=CANONICAL_DIFFICULTY_ORDER,
    )


# ============================================================================
# Single Skill Training (for debugging)
# ============================================================================


def train_single_skill(
    mission_cls: type[AtomicSkillMission],
    num_cogs: int = 1,
) -> TrainTool:
    """Train on a single atomic skill without curriculum.

    Useful for debugging or focused skill development.

    Args:
        mission_cls: Skill mission class to train on
        num_cogs: Number of agents

    Returns:
        TrainTool for single-skill training
    """
    env = make_atomic_skill_env(mission_cls=mission_cls, num_cogs=num_cogs)

    # Create single-env curriculum
    curriculum = cc.env_curriculum(env)

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    # Evaluate on the same skill
    mission_inst = mission_cls()  # type: ignore[call-arg]
    eval_suite = [
        SimulationConfig(
            suite="cvc_atomic_skills",
            name=f"{mission_inst.name}_{num_cogs}cogs",
            env=env,
        )
    ]

    evaluator_cfg = EvaluatorConfig(simulations=eval_suite, epoch_interval=10)
    checkpointer_cfg = CheckpointerConfig(epoch_interval=10)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        checkpointer=checkpointer_cfg,
    )
