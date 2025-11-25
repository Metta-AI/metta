"""ICL CoGs curriculum: Multi-agent In-Context Learning with increasing resource gathering requirements."""

from __future__ import annotations

import random
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.map_utils.resource_reducer import reduce_map_resources
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import AgentConfig, MettaGridConfig

# Resources required for a heart
REQUIRED_RESOURCES = ["carbon", "oxygen", "germanium", "silicon"]
RESOURCE_TO_CHAR = {
    "carbon": "C",
    "oxygen": "O",
    "germanium": "G",
    "silicon": "S",
}

class ICLMapVariant(MissionVariant):
    """Selects a map and filters extractors based on missing resources."""

    name: str = "icl_map"
    num_assemblers: int = 1
    missing_resources: tuple[str, ...] = ()
    size: str = "large"
    seed: int = 42

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        suffix = "_small" if self.size == "small" else ""
        map_filename = f"icl_cogs/{self.num_assemblers}_assembler{'s' if self.num_assemblers > 1 else ''}{suffix}.map"

        # We want to KEEP extractors for missing resources, and REMOVE extractors for present resources.
        # reduce_map_resources takes levels 0-10. 10=Keep, 0=Remove.

        resource_levels = {
            "&": 10, "+": 10, "=": 10, # Always keep buildings
        }

        for res in REQUIRED_RESOURCES:
            char = RESOURCE_TO_CHAR[res]
            if res in self.missing_resources:
                # Resource is missing from inventory -> Keep extractor on map
                resource_levels[char] = 10
            else:
                # Resource is present in inventory -> Remove extractor from map
                resource_levels[char] = 0

        reduced_map = reduce_map_resources(
            map_filename,
            resource_levels=resource_levels,
            seed=self.seed,
        )
        env.game.map_builder = reduced_map


class ICLInventoryVariant(MissionVariant):
    """Configures heterogeneous agent inventories."""

    name: str = "icl_inventory"
    missing_resources: tuple[str, ...] = ()
    seed: int = 42

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        rng = random.Random(self.seed)
        num_agents = env.game.num_agents

        # Resources available to be distributed (NOT missing)
        available_resources = [r for r in REQUIRED_RESOURCES if r not in self.missing_resources]

        # Create specific agent configs for heterogeneity
        # Start with a copy of the base agent config
        base_agent_config = env.game.agent.model_copy()

        agents = []
        for i in range(num_agents):
            agent_cfg = base_agent_config.model_copy()
            agent_cfg.initial_inventory = dict(base_agent_config.initial_inventory)
            agents.append(agent_cfg)

        # Distribute available resources among agents
        # Strategy: Ensure collectively they have what's needed.
        # For simplicity, give each available resource to a random agent (or all agents? The prompt says "one has X, other has Y")
        # Let's distribute them randomly.

        # If there are available resources, we want to ensure at least one set of them exists.
        # Actually, since they need to make hearts repeatedly (?), or just one?
        # "The agents are initialized with everything they need to operate an assembler to get a single heart"
        # So let's give them enough for 1 heart collectively.

        if available_resources:
            for res in available_resources:
                # Assign 1 unit of this resource to a random agent
                recipient = rng.choice(agents)
                recipient.initial_inventory[res] = recipient.initial_inventory.get(res, 0) + 1

        # Also, for the "Fully Initialized" case where they have EVERYTHING,
        # "Make sure that collectively they have what they need."
        # We just did that.

        env.game.agents = agents


def make_icl_curriculum(
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    size: str = "large",
) -> CurriculumConfig:

    all_mission_tasks = []

    # Iterate through difficulty levels
    # 1. Num Assemblers (1-4)
    # 2. Num Missing Resources (0-4)

    for num_assemblers in range(1, 5):
        # Create combinations of missing resources
        # Level 0: 0 missing (Full init)
        # Level 1: 1 missing
        # ...

        # To keep curriculum size manageable, we might not want EVERY permutation of missing resources.
        # But let's do:
        # - 0 missing (1 variant)
        # - 1 missing (4 variants)
        # - 2 missing (random selection? or just a few?)
        # - 4 missing (1 variant - standard game)

        # Let's stick to a progression of Count of missing resources.

        for num_missing in range(5): # 0, 1, 2, 3, 4
            # Pick a set of missing resources for this level.
            # If we want robust training, we should probably sample a few combinations or rotate them.
            # For a static curriculum definition, we need fixed variants.

            if num_missing == 0:
                missing_sets = [()]
            elif num_missing == 4:
                missing_sets = [tuple(REQUIRED_RESOURCES)]
            else:
                # Pick a few representative combinations
                # e.g. just rotate through them for num_missing=1
                if num_missing == 1:
                    missing_sets = [(r,) for r in REQUIRED_RESOURCES]
                elif num_missing == 2:
                    # Pairs: (C,O), (G,S) maybe?
                    missing_sets = [("carbon", "oxygen"), ("germanium", "silicon")]
                elif num_missing == 3:
                     missing_sets = [("carbon", "oxygen", "germanium"), ("oxygen", "germanium", "silicon")]

            for missing in missing_sets:
                # Construct Mission
                # We don't have a base Mission object for these custom maps in the registry,
                # so we define it on the fly or use a dummy base.

                # We can use a "template" mission just to satisfy the type system,
                # but we'll overwrite everything in the variants.

                name = f"icl_asm{num_assemblers}_miss{num_missing}_{'_'.join(r[0] for r in missing)}_{size}"

                variants = [
                    ICLMapVariant(num_assemblers=num_assemblers, missing_resources=missing, size=size),
                    ICLInventoryVariant(missing_resources=missing),
                ]

                # We use a dummy site because the MapVariant overwrites it.
                from cogames.cogs_vs_clips.sites import TRAINING_FACILITY

                mission = Mission(
                    name=name,
                    description=f"{num_assemblers} Assemblers, Missing {num_missing} resources",
                    site=TRAINING_FACILITY, # Placeholder
                    variants=variants,
                    num_cogs=num_cogs,
                )

                env = mission.make_env()
                env.label = name # Helpful for logging

                tasks = cc.bucketed(env)
                tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 1.0])

                all_mission_tasks.append(tasks)

    merged_tasks = cc.merge(all_mission_tasks)

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=2000,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return merged_tasks.to_curriculum(
        num_active_tasks=500,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    size: str = "large",
) -> TrainTool:
    """Train on ICL CoGs curriculum."""

    curriculum = curriculum or make_icl_curriculum(
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        size=size,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # Simple eval suite: just the max difficulty (4 missing, all assemblers)
    from cogames.cogs_vs_clips.sites import TRAINING_FACILITY
    eval_env = Mission(
        name="icl_eval_hard",
        description="Eval Hard",
        site=TRAINING_FACILITY,
        variants=[
            ICLMapVariant(num_assemblers=4, missing_resources=("carbon", "oxygen", "germanium", "silicon"), size=size),
            ICLInventoryVariant(missing_resources=("carbon", "oxygen", "germanium", "silicon"))
        ],
        num_cogs=num_cogs
    ).make_env()

    evaluator_cfg = EvaluatorConfig(
        simulations=[SimulationConfig(suite="icl", name="icl_hard", env=eval_env)],
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
    )

def play(
    num_assemblers: int = 1,
    missing_count: int = 0,
    num_cogs: int = 4,
    policy_uri: Optional[str] = None,
    size: str = "large",
    agent_policy: str = "noop",
) -> PlayTool:
    """Play a specific configuration."""

    # Determine missing resources based on count
    if missing_count == 0:
        missing = ()
    elif missing_count == 4:
        missing = tuple(REQUIRED_RESOURCES)
    else:
        # Just take the first N for simplicity in play tool
        missing = tuple(REQUIRED_RESOURCES[:missing_count])

    from cogames.cogs_vs_clips.sites import TRAINING_FACILITY
    mission = Mission(
        name="icl_play",
        description="Play Mode",
        site=TRAINING_FACILITY,
        variants=[
            ICLMapVariant(num_assemblers=num_assemblers, missing_resources=missing, size=size),
            ICLInventoryVariant(missing_resources=missing)
        ],
        num_cogs=num_cogs
    )

    env = mission.make_env()

    sim = SimulationConfig(
        suite="icl_play",
        name=f"asm{num_assemblers}_miss{missing_count}_{size}",
        env=env
    )

    return PlayTool(sim=sim, policy_uri=policy_uri, policy_type=agent_policy)

def evaluate(
    policy_uris: str | Sequence[str],
    num_cogs: int = 4,
    size: str = "large",
) -> EvaluateTool:
    """Evaluate on the hardest configuration."""
    from cogames.cogs_vs_clips.sites import TRAINING_FACILITY
    eval_env = Mission(
        name="icl_eval_hard",
        description="Eval Hard",
        site=TRAINING_FACILITY,
        variants=[
            ICLMapVariant(num_assemblers=4, missing_resources=("carbon", "oxygen", "germanium", "silicon"), size=size),
            ICLInventoryVariant(missing_resources=("carbon", "oxygen", "germanium", "silicon"))
        ],
        num_cogs=num_cogs
    ).make_env()

    return EvaluateTool(
        simulations=[SimulationConfig(suite="icl", name="icl_hard", env=eval_env)],
        policy_uris=policy_uris
    )
