import random
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Field,
    Position,
    ChestConfig,
    RecipeConfig,
)
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from cogames.cogs_vs_clips.scenarios import make_game
from metta.tools.play import PlayTool
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.loss import LossConfig
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from mettagrid.mapgen.mapgen import MapGen
from metta.map.terrain_from_numpy import CogsVClippiesFromNumpy
from metta.tools.replay import ReplayTool
from experiments.recipes.cogs_v_clips.config import (
    generalized_terrain_curriculum_args,
    obj_distribution_by_room_size,
)


class GeneralizedTerrainTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["GeneralizedTerrainTaskGenerator"]):
        num_cogs: list[int] = Field(default=[1])
        num_obj_distribution: list[int] = Field(default=[10])
        positions: list[list[Position]] = Field(default=[["Any"]])
        regeneration_rate: list[int] = Field(default=[5])
        sizes: list[str] = Field(default=["small"])
        use_base: list[bool] = Field(default=[True])
        heart_reward: list[int] = Field(default=[0, 1, 2])
        heart_in_chest_reward: list[int] = Field(default=[3, 5, 10])
        resource_rewards: list[float] = Field(default=[0, 0.5, 1.0])

    def __init__(self, config: "GeneralizedTerrainTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _overwrite_positions(self, object):
        for i, recipe in enumerate(object.recipes):
            object.recipes[i] = (["Any"], recipe[1])

    def configure_env_agent(self, env, rng):
        """Configure parameters for agent, such as reward"""
        env.game.inventory_regen_interval = rng.choice(self.config.regeneration_rate)
        env.game.agent.inventory_regen_amounts = {"energy": 1}
        env.game.agent.shareable_resources = ["energy"]
        env.game.agent.rewards.stats = {
            "chest.heart.amount": rng.choice(self.config.heart_in_chest_reward)
        }

        heart_reward = rng.choice(self.config.heart_reward)

        env.game.agent.rewards.inventory = {
            "heart": heart_reward,
            "carbon": rng.choice(self.config.resource_rewards),
            "oxygen": rng.choice(self.config.resource_rewards),
            "germanium": rng.choice(self.config.resource_rewards),
            "silicon": rng.choice(self.config.resource_rewards),
        }

        # if there are no rewards for hearts, soemtimes initialize agents with hearts in inventory
        if heart_reward == 0:
            env.game.agent.initial_inventory = {
                "heart": rng.choice([0, 1, 2, 3]),
                "energy": 100,
            }

    def setup_map_builder(self, num_cogs, room_size, rng):
        """Make the mapbuilder, which takes terrain and populates with objects"""
        num_obj_distribution = obj_distribution_by_room_size[room_size]

        num_assemblers = rng.choice(num_obj_distribution)
        num_chargers = rng.choice(num_obj_distribution)
        num_chests = max(4, rng.choice(num_obj_distribution))

        if rng.choice([True, False]):
            num_extractors = {
                "carbon": rng.choice([0] + num_obj_distribution),
                "oxygen": rng.choice([0] + num_obj_distribution),
                "germanium": rng.choice([0] + num_obj_distribution),
                "silicon": rng.choice([0] + num_obj_distribution),
            }
        else:
            num_extractors = {"carbon": 0, "oxygen": 0, "germanium": 0, "silicon": 0}

        env = make_game(
            num_cogs=num_cogs,
            num_assemblers=num_assemblers,
            num_chargers=num_chargers,
            num_carbon_extractors=num_extractors["carbon"],
            num_oxygen_extractors=num_extractors["oxygen"],
            num_germanium_extractors=num_extractors["germanium"],
            num_silicon_extractors=num_extractors["silicon"],
            num_chests=num_chests,
        )
        map_builder = MapGen.Config(
            instances=24 // num_cogs,
            border_width=6,
            instance_border_width=3,
            instance=CogsVClippiesFromNumpy.Config(
                agents=num_cogs,
                objects={
                    "assembler": num_assemblers,
                    "charger": num_chargers,
                    "carbon_extractor": num_extractors["carbon"],
                    "oxygen_extractor": num_extractors["oxygen"],
                    "germanium_extractor": num_extractors["germanium"],
                    "silicon_extractor": num_extractors["silicon"],
                    "chest": num_chests,
                },
                remove_altars=True,
                dir=f"varied_terrain/{rng.choice(['sparse', 'balanced', 'dense'])}_{room_size}",
                mass_in_center=rng.choice(self.config.use_base),
                rng=rng,
            ),
        )
        env.game.map_builder = map_builder
        env.game.num_agents = 24

        return env, num_extractors

    def _make_env_cfg(self, rng: random.Random):
        num_cogs = rng.choice(self.config.num_cogs)
        position = rng.choice(self.config.positions)
        room_size = rng.choice(self.config.sizes)
        env, num_extractors = self.setup_map_builder(num_cogs, room_size, rng)

        self._make_assembler_recipes(
            env.game.objects["assembler"], rng, num_extractors, position
        )

        for obj in [
            "charger",
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        ]:
            # for extractors and chargers, any agent can use
            self._overwrite_positions(env.game.objects[obj])

        env.game.objects["chest"] = ChestConfig(
            type_id=17,
            resource_type="heart",
            position_deltas=[("N", 1), ("S", 1), ("E", 1), ("W", 1)],
        )

        self.configure_env_agent(env, rng)

        env.label = f"{env.game.num_agents}_cogs_{room_size}_{env.game.inventory_regen_interval}_regeneration_rate"

        return env

    def _make_assembler_recipes(
        self,
        assembler,
        rng: random.Random,
        num_extractors: dict[str, int],
        assembler_position: list[Position],
    ):
        input_resources = {"energy": 3}
        num_input_resources = 0
        for resource, num_extractor in num_extractors.items():
            if (
                num_extractor > 0
                and rng.choice([True, False])
                and num_input_resources <= 2
            ):
                input_resources[resource] = 1
                num_input_resources += 1
        assembler.recipes = [
            (
                assembler_position,
                RecipeConfig(
                    input_resources=input_resources,
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )
        ]
        return assembler

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        env = self._make_env_cfg(rng)

        return env

    def generate_task(self, task_id: int, rng: random.Random, num_instances: Optional[int] = None) -> MettaGridConfig:
        return self._generate_task(task_id, rng)


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "multi_agent_pairs") -> PlayTool:
    task_generator = GeneralizedTerrainTaskGenerator(
        config=GeneralizedTerrainTaskGenerator.Config(
            **generalized_terrain_curriculum_args[curriculum_style]
        )
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="cogs_vs_clippies", name="play"
        )
    )


def train(
    curriculum_style: str = "multi_agent_pairs_uniform", architecture="vit_reset"
) -> TrainTool:
    from experiments.evals.cogs_v_clips import make_cogs_v_clips_evals

    task_generator_cfg = GeneralizedTerrainTaskGenerator.Config(
        **generalized_terrain_curriculum_args[curriculum_style]
    )
    algorithm_config = LearningProgressConfig(
        ema_timescale=0.001,
        exploration_bonus=0.15,
        max_memory_tasks=1000,
        max_slice_axes=3,
        progress_smoothing=0.15,
        num_active_tasks=1000,
        rand_task_rate=0.25,
    )
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    if architecture == "vit_reset":
        policy_config = ViTResetConfig()
    elif architecture == "lstm_reset":
        policy_config = FastLSTMResetConfig()

    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=make_cogs_v_clips_evals(),
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    position=["Any"],
    regeneration_rate=2,
    use_base=True,
    sizes="small",
):
    task_generator = GeneralizedTerrainTaskGenerator(
        config=GeneralizedTerrainTaskGenerator.Config(
            num_cogs=[num_cogs],
            positions=[position],
            regeneration_rate=[regeneration_rate],
            use_base=[use_base],
            sizes=[sizes],
        )
    )
    return task_generator.get_task(random.randint(0, 1000000))


def replay() -> ReplayTool:
    eval_env = make_env(
        num_cogs=2,
        sizes="small",
        regeneration_rate=2,
        position=["Any", "Any"],
        use_base=True,
    )
    policy_uri = "s3://softmax-public/policies/cogs_v_clips.level_1.eval_local.multi_agent_pairs_bases_vit_reset.2025-10-02/:latest"

    return ReplayTool(
        policy_uri=policy_uri,
        sim=SimulationConfig(suite="cogs_v_clips", env=eval_env, name="eval"),
    )


def experiment():
    import subprocess
    import time

    for curriculum_style in generalized_terrain_curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.cogs_v_clips.generalized_terrain.train",
                f"run=cogs_v_clips.generalized_terrain.{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
