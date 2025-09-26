from metta.agent.policies.fast import FastConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.tools.train import TrainTool


def test_policy_alias_fast_resolves():
    tool = TrainTool(policy_architecture="fast")
    assert isinstance(tool.policy_architecture, FastConfig)


def test_policy_alias_transformer_resolves():
    tool = TrainTool(policy_architecture="transformer")
    assert isinstance(tool.policy_architecture, TransformerPolicyConfig)
