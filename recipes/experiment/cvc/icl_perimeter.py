"""ICL Perimeter curriculum: In-Context Learning with perimeter-based maps.

This curriculum teaches agents to make hearts incrementally:
- Agents start with most resources in their inventory (4× heart cost)
- Missing resources must be gathered from extractors on the map perimeter
- Extractors give exactly 1 heart's worth per use (single use)

Map structure uses ICLPerimeterMapBuilder:
- Outer walkable border (agents can surround objects)
- Inner object border (extractors, assembler, chest, charger)
- Interior where agents spawn

Curriculum progression:
- 0 missing: All resources in inventory, just craft and deposit
- 1 missing: Gather one resource type
- 2 missing: Gather two resource types
- 3 missing: Gather three resource types
- 4 missing: Full game, gather all resources
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.sites import Site
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.icl_perimeter import ICLPerimeterMapBuilder

# Resources required for a heart
REQUIRED_RESOURCES = ["carbon", "oxygen", "germanium", "silicon"]

# Map resource names to extractor object names
RESOURCE_TO_EXTRACTOR = {
    "carbon": "carbon_extractor",
    "oxygen": "oxygen_extractor",
    "germanium": "germanium_extractor",
    "silicon": "silicon_extractor",
}


class ICLPerimeterMapVariant(MissionVariant):
    """Creates a perimeter-based map with only the needed extractors."""

    name: str = "icl_perimeter_map"
    missing_resources: tuple[str, ...] = ()
    num_agents: int = 4
    map_width: int = 15
    map_height: int = 15
    seed: int = 42

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        # Build objects dict: only include extractors for missing resources
        objects: dict[str, int] = {
            "assembler": 1,
            "chest": 1,
            "charger": 1,
        }

        # Add extractors only for missing resources
        for res in self.missing_resources:
            extractor_name = RESOURCE_TO_EXTRACTOR[res]
            objects[extractor_name] = 1

        # Create the map builder config
        map_builder = ICLPerimeterMapBuilder.Config(
            width=self.map_width,
            height=self.map_height,
            objects=objects,
            num_agents=self.num_agents,
            seed=self.seed,
            border_width=1,  # Wall border
            border_object="wall",
            outer_walkable_width=1,  # Walkable border for surrounding objects
            density="no-terrain",  # No interior obstacles
        )

        env.game.map_builder = map_builder


class ICLInventoryVariant(MissionVariant):
    """Configures agent inventories with available resources at Nx heart cost."""

    name: str = "icl_inventory"
    missing_resources: tuple[str, ...] = ()
    heart_multiplier: int = 4  # Give agents this many hearts worth of each resource

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        # Resources available (NOT missing) will be given to all agents
        available_resources = [r for r in REQUIRED_RESOURCES if r not in self.missing_resources]

        if not available_resources:
            return

        # Calculate resource amounts based on heart cost
        heart_cost = mission.assembler.first_heart_cost
        per_heart = {
            "carbon": heart_cost,
            "oxygen": heart_cost,
            "germanium": max(1, heart_cost // 5),
            "silicon": 3 * heart_cost,
        }

        # Update the base agent config - all agents get the same inventory
        agent_cfg = env.game.agent
        agent_cfg.initial_inventory = dict(agent_cfg.initial_inventory)

        def _limit_for(resource: str) -> int:
            return agent_cfg.get_limit_for_resource(resource)

        for res in available_resources:
            target_amount = per_heart[res] * self.heart_multiplier
            cap = _limit_for(res)
            current = int(agent_cfg.initial_inventory.get(res, 0))
            agent_cfg.initial_inventory[res] = min(cap, current + target_amount)


class ICLExtractorVariant(MissionVariant):
    """Configures extractors for missing resources to give exactly one heart's worth per use."""

    name: str = "icl_extractor"
    missing_resources: tuple[str, ...] = ()
    max_uses: int = 0  # Unlimited uses
    cooldown: int = 10  # Cooldown between uses

    def modify_mission(self, mission: Mission) -> None:
        if not self.missing_resources:
            return

        # Calculate what each extractor needs to output for one heart
        heart_cost = mission.assembler.first_heart_cost
        per_heart = {
            "carbon": heart_cost,  # default 10
            "oxygen": heart_cost,  # default 10
            "germanium": max(1, heart_cost // 5),  # default 2
            "silicon": 3 * heart_cost,  # default 30
        }

        # Default extractor outputs (at efficiency=100):
        # Carbon: 2 per use
        # Oxygen: 10 per use (fixed, efficiency affects cooldown)
        # Germanium: 2 per use (fixed, efficiency affects cooldown)
        # Silicon: 15 per use

        # Calculate efficiency to get exactly one heart's worth per use
        # Extractors are unlimited with a short cooldown
        if "carbon" in self.missing_resources:
            efficiency = (per_heart["carbon"] * 100) // 2  # 2 is default output
            mission.carbon_extractor = CarbonExtractorConfig(
                efficiency=min(500, efficiency), max_uses=self.max_uses
            )

        if "oxygen" in self.missing_resources:
            mission.oxygen_extractor = OxygenExtractorConfig(
                efficiency=100, max_uses=self.max_uses
            )

        if "germanium" in self.missing_resources:
            mission.germanium_extractor = GermaniumExtractorConfig(
                efficiency=100, max_uses=self.max_uses
            )

        if "silicon" in self.missing_resources:
            efficiency = (per_heart["silicon"] * 100) // 15  # 15 is default output
            mission.silicon_extractor = SiliconExtractorConfig(
                efficiency=min(500, efficiency), max_uses=self.max_uses
            )


class ICLCargoCapacityVariant(MissionVariant):
    """Sets cargo capacity to maximum (255) to allow storing multiple hearts worth of resources."""

    name: str = "icl_cargo_capacity"
    cargo_capacity: int = 255

    def modify_mission(self, mission: Mission) -> None:
        mission.cargo_capacity = self.cargo_capacity


class ICLChestVariant(MissionVariant):
    """Configures chest vibe transfers - agents can deposit hearts with any resource vibe."""

    name: str = "icl_chest"

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        from mettagrid.config.mettagrid_config import ChestConfig

        # Get the chest config and set vibe transfers for all vibes
        chest_cfg = env.game.objects.get("chest")
        if not isinstance(chest_cfg, ChestConfig):
            return
        chest_cfg.vibe_transfers = {
            "default": {"heart": 1},
            "heart_a": {"heart": 1},
            "heart_b": {"heart": 1},
            "carbon_a": {"heart": 1},
            "carbon_b": {"heart": 1},
            "oxygen_a": {"heart": 1},
            "oxygen_b": {"heart": 1},
            "germanium_a": {"heart": 1},
            "germanium_b": {"heart": 1},
            "silicon_a": {"heart": 1},
            "silicon_b": {"heart": 1},
        }


class ICLAssemblerVariant(MissionVariant):
    """Configures assembler to accept multiple vibe options for heart assembly."""

    name: str = "icl_assembler"

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        from mettagrid.config.mettagrid_config import AssemblerConfig, ProtocolConfig

        assembler_cfg = env.game.objects.get("assembler")
        if not isinstance(assembler_cfg, AssemblerConfig):
            return

        # Get heart cost from mission
        heart_cost = mission.assembler.first_heart_cost
        additional_cost = mission.assembler.additional_heart_cost

        # Define all vibes that can be used for heart assembly
        heart_vibes = ["heart_a", "heart_b", "carbon_a", "carbon_b", "oxygen_a", "oxygen_b",
                       "germanium_a", "germanium_b", "silicon_a", "silicon_b"]

        # Build protocols for each vibe option and each agent count
        new_protocols = []
        for vibe in heart_vibes:
            for i in range(4):  # 1-4 agents
                new_protocols.append(
                    ProtocolConfig(
                        vibes=[vibe] * (i + 1),
                        input_resources={
                            "carbon": heart_cost + additional_cost * i,
                            "oxygen": heart_cost + additional_cost * i,
                            "germanium": max(1, (heart_cost + additional_cost * i) // 5),
                            "silicon": 3 * (heart_cost + additional_cost * i),
                        },
                        output_resources={"heart": i + 1},
                    )
                )

        assembler_cfg.protocols = new_protocols

class ICLVibeVariant(MissionVariant):
    """Restricts action space to first N vibes."""

    name: str = "icl_vibe"
    vibe_count: int = 13  # First 13 vibes only

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        from mettagrid.config import vibes

        # Modify vibe_names and number_of_vibes AFTER config creation
        # (setting vibe_count in modify_mission would cause validation error)
        first_n_vibes = [v.name for v in vibes.VIBES[: self.vibe_count]]
        env.game.vibe_names = first_n_vibes
        if env.game.actions.change_vibe:
            env.game.actions.change_vibe.number_of_vibes = self.vibe_count


# Create a simple site for ICL missions
ICL_SITE = Site(
    name="icl_perimeter",
    description="ICL Perimeter Training Site",
    min_cogs=1,
    max_cogs=8,
    map_builder=ICLPerimeterMapBuilder.Config(),  # Default, will be overwritten by variant
)


def make_icl_curriculum(
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create ICL curriculum with perimeter maps.

    Args:
        num_cogs: Number of agents per environment
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration

    Returns:
        CurriculumConfig with learning progress algorithm
    """
    all_mission_tasks = []

    # Curriculum: vary number of missing resources (0 to 4)
    for num_missing in range(5):  # 0, 1, 2, 3, 4
        if num_missing == 0:
            missing_sets = [()]
        elif num_missing == 4:
            missing_sets = [tuple(REQUIRED_RESOURCES)]
        else:
            # Create representative combinations
            if num_missing == 1:
                missing_sets = [(r,) for r in REQUIRED_RESOURCES]
            elif num_missing == 2:
                missing_sets = [("carbon", "oxygen"), ("germanium", "silicon")]
            elif num_missing == 3:
                missing_sets = [("carbon", "oxygen", "germanium"), ("oxygen", "germanium", "silicon")]

        for missing in missing_sets:
            name = f"icl_miss{num_missing}_{'_'.join(r[0] for r in missing) if missing else 'none'}"

            variants = [
                ICLCargoCapacityVariant(),
                ICLVibeVariant(),
                ICLChestVariant(),
                ICLAssemblerVariant(),
                ICLPerimeterMapVariant(
                    missing_resources=missing,
                    num_agents=num_cogs,
                ),
                ICLInventoryVariant(missing_resources=missing),
                ICLExtractorVariant(missing_resources=missing),
            ]

            mission = Mission(
                name=name,
                description=f"ICL: {num_missing} missing resources",
                site=ICL_SITE,
                variants=variants,
                num_cogs=num_cogs,
            )

            env = mission.make_env()
            env.label = name

            tasks = cc.bucketed(env)
            # tasks.add_bucket("game.agent.rewards.inventory.heart", [0.0, 0.1, 1.0])
            tasks.add_bucket("game.map_builder.objects.assembler", [1, 2, 3])
            tasks.add_bucket("game.map_builder.width", [8, 10, 12, 14, 16, 18, 20])
            tasks.add_bucket("game.map_builder.height", [8, 10, 12, 14, 16, 18, 20])
            tasks.add_bucket("game.max_steps", [250, 350, 500, 600])

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
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 4,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train on ICL perimeter curriculum."""
    curriculum = curriculum or make_icl_curriculum(
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # Eval on hardest config (all 4 missing)
    eval_mission = Mission(
        name="icl_eval_hard",
        description="ICL Eval: All resources missing",
        site=ICL_SITE,
        variants=[
            ICLCargoCapacityVariant(),
            ICLVibeVariant(),
            ICLChestVariant(),
            ICLAssemblerVariant(),
            ICLPerimeterMapVariant(
                missing_resources=tuple(REQUIRED_RESOURCES),
                num_agents=num_cogs,
            ),
            ICLInventoryVariant(missing_resources=tuple(REQUIRED_RESOURCES)),
            ICLExtractorVariant(missing_resources=tuple(REQUIRED_RESOURCES)),
        ],
        num_cogs=num_cogs,
    )
    eval_env = eval_mission.make_env()

    evaluator_cfg = EvaluatorConfig(
        simulations=[SimulationConfig(suite="icl", name="icl_hard", env=eval_env)],
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
    )


def play(
    missing_count: int = 0,
    num_cogs: int = 4,
    policy_uri: Optional[str] = None,
) -> PlayTool:
    """Play a specific ICL configuration.

    Args:
        missing_count: Number of missing resources (0-4)
        num_cogs: Number of agents
        policy_uri: Optional policy URI to load
    """
    if missing_count == 0:
        missing = ()
    elif missing_count == 4:
        missing = tuple(REQUIRED_RESOURCES)
    else:
        missing = tuple(REQUIRED_RESOURCES[:missing_count])

    mission = Mission(
        name="icl_play",
        description="ICL Play Mode",
        site=ICL_SITE,
        variants=[
            ICLCargoCapacityVariant(),
            ICLVibeVariant(),
            ICLChestVariant(),
            ICLAssemblerVariant(),
            ICLPerimeterMapVariant(
                missing_resources=missing,
                num_agents=num_cogs,
            ),
            ICLInventoryVariant(missing_resources=missing),
            ICLExtractorVariant(missing_resources=missing),
        ],
        num_cogs=num_cogs,
    )

    env = mission.make_env()
    sim = SimulationConfig(suite="icl_play", name=f"miss{missing_count}", env=env)

    return PlayTool(sim=sim, policy_uri=policy_uri)


def evaluate(
    policy_uris: str | Sequence[str],
    num_cogs: int = 4,
) -> EvaluateTool:
    """Evaluate on the hardest ICL configuration (all resources missing)."""
    eval_mission = Mission(
        name="icl_eval_hard",
        description="ICL Eval: All resources missing",
        site=ICL_SITE,
        variants=[
            ICLCargoCapacityVariant(),
            ICLVibeVariant(),
            ICLChestVariant(),
            ICLAssemblerVariant(),
            ICLPerimeterMapVariant(
                missing_resources=tuple(REQUIRED_RESOURCES),
                num_agents=num_cogs,
            ),
            ICLInventoryVariant(missing_resources=tuple(REQUIRED_RESOURCES)),
            ICLExtractorVariant(missing_resources=tuple(REQUIRED_RESOURCES)),
        ],
        num_cogs=num_cogs,
    )
    eval_env = eval_mission.make_env()

    return EvaluateTool(
        simulations=[SimulationConfig(suite="icl", name="icl_hard", env=eval_env)],
        policy_uris=policy_uris,
    )


def experiment(
    run_name: Optional[str] = None,
    num_cogs: int = 1,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit an ICL perimeter training job on AWS with 4 GPUs.

    Args:
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per environment (default: 4)
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600)
        skip_git_check: Whether to skip git check (default: True)
        additional_args: Additional arguments to pass to the training command

    Examples:
        Submit training:
            uv run ./tools/run.py recipes.experiment.cvc.icl_perimeter.experiment

        Submit with custom name:
            uv run ./tools/run.py recipes.experiment.cvc.icl_perimeter.experiment \\
                run_name=icl_perimeter_test
    """
    if run_name is None:
        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        run_name = f"george.icl_perimeter_{num_cogs}cogs_{timestamp}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.icl_perimeter.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        f"--heartbeat-timeout-seconds={heartbeat_timeout}",
    ]

    if skip_git_check:
        cmd.append("--skip-git-check")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching ICL perimeter training job: {run_name}")
    print(f"  Agents: {num_cogs}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


if __name__ == "__main__":
    #single agent
    experiment(num_cogs=1)

    #2 agents
    experiment(num_cogs=2)

    #4 agents
    experiment(num_cogs=4)

