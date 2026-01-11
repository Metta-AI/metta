"""Arena recipe with CMPO (Conservative Model-Based Policy Optimization)."""

from __future__ import annotations

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.loss.cmpo import CMPOConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from recipes.experiment.arena import make_curriculum, simulations
from recipes.experiment.arena import train_shaped as base_train_shaped
from recipes.prod.arena_basic_easy_shaped import train as arena_basic_easy_shaped_train
import metta.tools as tools

def cmpo_losses() -> LossesConfig:
    losses = LossesConfig()
    losses.ppo_actor.enabled = False
    losses.ppo_critic.enabled = False
    losses.cmpo = CMPOConfig(enabled=True)
    return losses


def _cmpo_trainer_config() -> TrainerConfig:
    return TrainerConfig(losses=cmpo_losses())


def _with_cmpo(base_tool: tools.TrainTool) -> tools.TrainTool:
    return tools.TrainTool(
        training_env=base_tool.training_env,
        trainer=_cmpo_trainer_config(),
        evaluator=base_tool.evaluator,
        policy_architecture=ViTDefaultConfig(),
    )


def train(enable_detailed_slice_logging: bool = False) -> tools.TrainTool:
    curriculum = make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)
    trainer_config = _cmpo_trainer_config()

    return tools.TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=ViTDefaultConfig(),
    )


def train_shaped(rewards: bool = True) -> tools.TrainTool:
    base_tool = base_train_shaped(rewards=rewards)
    return _with_cmpo(base_tool)


def basic_easy_shaped() -> tools.TrainTool:
    base_tool = arena_basic_easy_shaped_train()
    return _with_cmpo(base_tool)
