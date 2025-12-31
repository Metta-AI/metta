"""Arena recipe with CMPO (Conservative Model-Based Policy Optimization)."""

from typing import Any

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.nodes.cmpo import CMPOConfig
from metta.rl.nodes import default_nodes
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from recipes.experiment.arena import make_curriculum, simulations
from recipes.experiment.arena import train_shaped as base_train_shaped
from recipes.prod.arena_basic_easy_shaped import train as arena_basic_easy_shaped_train


def cmpo_nodes() -> dict[str, Any]:
    nodes = default_nodes()
    nodes["ppo_actor"].enabled = False
    nodes["ppo_critic"].enabled = False
    nodes["cmpo"] = CMPOConfig(enabled=True)
    return nodes


def _cmpo_trainer_config() -> TrainerConfig:
    return TrainerConfig(nodes=cmpo_nodes())


def _with_cmpo(base_tool: TrainTool) -> TrainTool:
    return TrainTool(
        training_env=base_tool.training_env,
        trainer=_cmpo_trainer_config(),
        evaluator=base_tool.evaluator,
        policy_architecture=ViTDefaultConfig(),
    )


def train(enable_detailed_slice_logging: bool = False) -> TrainTool:
    curriculum = make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)
    trainer_config = _cmpo_trainer_config()

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=ViTDefaultConfig(),
    )


def train_shaped(rewards: bool = True) -> TrainTool:
    base_tool = base_train_shaped(rewards=rewards)
    return _with_cmpo(base_tool)


def basic_easy_shaped() -> TrainTool:
    base_tool = arena_basic_easy_shaped_train()
    return _with_cmpo(base_tool)
