from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
from torchrl.data import Composite

from experiments.recipes.losses import grpo as grpo_recipe
from metta.agent.policies.vit_grpo import ViTGRPOConfig
from metta.agent.policy import Policy
from metta.rl.loss.grpo import GRPO, GRPOConfig
from metta.rl.training import TrainingEnvironment
from metta.tools.train import TrainTool


@pytest.mark.parametrize(
    "factory",
    [
        grpo_recipe.train,
        grpo_recipe.train_shaped,
        grpo_recipe.basic_easy_shaped,
    ],
)
def test_grpo_recipes_configure_grpo_loss(factory: Callable[[], TrainTool]) -> None:
    tool = factory()

    assert isinstance(tool, TrainTool)

    loss_configs = tool.trainer.losses.loss_configs
    assert set(loss_configs) == {"grpo"}

    grpo_config = loss_configs["grpo"]
    assert isinstance(grpo_config, GRPOConfig)

    policy = MagicMock(spec=Policy)
    policy.get_agent_experience_spec.return_value = Composite()
    policy.reset_memory.return_value = None
    policy.device = torch.device("cpu")
    policy.parameters.return_value = []

    env = MagicMock(spec=TrainingEnvironment)

    loss_instance = grpo_config.create(
        policy=policy,
        trainer_cfg=tool.trainer,
        env=env,
        device=torch.device("cpu"),
        instance_name="grpo",
        loss_config=grpo_config,
    )

    assert isinstance(loss_instance, GRPO)
    assert isinstance(tool.policy_architecture, ViTGRPOConfig)
