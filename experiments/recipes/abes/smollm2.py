"""Arena training recipes tailored for the SmolLM2 policy."""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.smollm2 import SmolLM2Config
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from mettagrid import MettaGridConfig
from metta.rl.loss import LossConfig, PPOConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_mettagrid(num_agents: int = 12) -> MettaGridConfig:
    """Create a smaller arena environment with easy shaped rewards for LLM training."""
    arena_env = eb.make_arena(num_agents=num_agents)
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}
    return arena_env


def make_tiny_mettagrid(num_agents: int = 6) -> MettaGridConfig:
    """Create a tiny arena environment for CPU debugging with easy shaped rewards."""
    arena_env = eb.make_arena(num_agents=num_agents)
    arena_env.game.map_builder.width = 12
    arena_env.game.map_builder.height = 12

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}
    return arena_env


def make_curriculum(arena_env: Optional[MettaGridConfig] = None) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()
    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    return CurriculumConfig(task_generator=arena_tasks)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena_smollm2/basic", env=basic_env),
        SimulationConfig(name="arena_smollm2/combat", env=combat_env),
    ]


def _default_trainer_config(curriculum: Optional[CurriculumConfig]) -> TrainerConfig:
    return TrainerConfig(
        losses=LossConfig(
            loss_configs={
                "ppo": PPOConfig(
                    clip_coef=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                )
            }
        ),
        curriculum=curriculum or make_curriculum(),
        batch_size=4096,
        minibatch_size=128,
        bptt_horizon=4,
        forward_pass_minibatch_target_size=16,
        update_epochs=8,
        async_factor=1,
        compile=False,
        evaluation=EvaluationConfig(
            simulations=make_evals(),
            evaluate_interval=100,
        ),
    )


def train(
    curriculum: Optional[CurriculumConfig] = None, freeze_llm: bool = True
) -> TrainTool:
    policy_cfg = SmolLM2Config(freeze_llm=freeze_llm)
    trainer_cfg = _default_trainer_config(curriculum)
    return TrainTool(trainer=trainer_cfg, policy_architecture=policy_cfg)


def train_frozen() -> TrainTool:
    return train(freeze_llm=True)


def train_unfrozen() -> TrainTool:
    return train(freeze_llm=False)


def train_high_throughput(
    curriculum: Optional[CurriculumConfig] = None,
    freeze_llm: bool = True,
) -> TrainTool:
    policy_cfg = SmolLM2Config(freeze_llm=freeze_llm)
    trainer_cfg = TrainerConfig(
        losses=LossConfig(
            loss_configs={
                "ppo": PPOConfig(
                    clip_coef=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                )
            }
        ),
        curriculum=curriculum or make_curriculum(),
        batch_size=8192,
        minibatch_size=256,
        bptt_horizon=8,
        forward_pass_minibatch_target_size=32,
        update_epochs=4,
        async_factor=1,
        compile=False,
        evaluation=EvaluationConfig(
            simulations=make_evals(),
            evaluate_interval=100,
        ),
    )
    return TrainTool(trainer=trainer_cfg, policy_architecture=policy_cfg)


def train_cpu_debug(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    policy_cfg = SmolLM2Config(freeze_llm=True)
    trainer_cfg = TrainerConfig(
        losses=LossConfig(
            loss_configs={
                "ppo": PPOConfig(
                    clip_coef=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                )
            }
        ),
        curriculum=curriculum or make_curriculum(make_tiny_mettagrid()),
        batch_size=256,
        minibatch_size=64,
        bptt_horizon=2,
        forward_pass_minibatch_target_size=8,
        update_epochs=4,
        async_factor=1,
        compile=False,
        evaluation=EvaluationConfig(
            simulations=make_evals(make_tiny_mettagrid()),
            evaluate_interval=200,
        ),
    )
    return TrainTool(trainer=trainer_cfg, policy_architecture=policy_cfg)


def evaluate(
    policy_uri: str, simulations: Optional[List[SimulationConfig]] = None
) -> SimTool:
    return SimTool(simulations=simulations or make_evals(), policy_uris=[policy_uri])


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))
