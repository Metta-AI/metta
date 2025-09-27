import pytest

from metta.agent.policies.fast import FastConfig
from metta.tools.train import TrainTool


def test_policy_alias_fast_resolves():
    tool = TrainTool(policy_architecture="fast")
    assert isinstance(tool.policy_architecture, FastConfig)


def test_policy_alias_transformer_not_supported():
    with pytest.raises(ValueError):
        TrainTool(policy_architecture="transformer")
