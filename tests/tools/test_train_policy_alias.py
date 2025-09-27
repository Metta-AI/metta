import pytest

from metta.agent.policies.agalite import AGaLiTeConfig, AGaLiTeImprovedConfig
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import TrainingEnvironmentConfig
from metta.tools.train import TrainTool


def _make_tool(**kwargs) -> TrainTool:
    base = {
        "trainer": TrainerConfig(),
        "training_env": TrainingEnvironmentConfig(),
    }
    base.update(kwargs)
    return TrainTool(**base)


def test_policy_alias_fast_resolves() -> None:
    tool = _make_tool(policy_architecture="fast")
    assert isinstance(tool.policy_architecture, FastConfig)


def test_policy_alias_agalite_resolves() -> None:
    tool = _make_tool(policy_architecture="agalite")
    assert isinstance(tool.policy_architecture, AGaLiTeConfig)


def test_policy_alias_agalite_improved_resolves() -> None:
    tool = _make_tool(policy_architecture="agalite_improved")
    assert isinstance(tool.policy_architecture, AGaLiTeImprovedConfig)


def test_policy_alias_transformer_resolves() -> None:
    tool = _make_tool(policy_architecture="transformer")
    assert isinstance(tool.policy_architecture, TransformerPolicyConfig)


def test_policy_alias_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown policy preset"):
        _make_tool(policy_architecture="unknown")
