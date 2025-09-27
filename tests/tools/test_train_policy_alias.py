from metta.agent.policies.fast import FastConfig
import pytest

from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.tools.train import TrainTool


def test_policy_alias_fast_resolves():
    tool = TrainTool(policy_architecture="fast", training_env=TrainingEnvironmentConfig())
    assert isinstance(tool.policy_architecture, FastConfig)


def test_policy_alias_transformer_not_supported():
    with pytest.raises(ValueError):
        TrainTool(policy_architecture="transformer", training_env=TrainingEnvironmentConfig())
