from metta.agent.policies.agalite import AGaLiTeConfig, AGaLiTeImprovedConfig
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.tools.train import TrainTool


def test_policy_alias_fast_resolves():
    tool = TrainTool(policy_architecture="fast")
    assert isinstance(tool.policy_architecture, FastConfig)


def test_policy_alias_agalite_resolves():
    tool = TrainTool(policy_architecture="agalite")
    assert isinstance(tool.policy_architecture, AGaLiTeConfig)


def test_policy_alias_agalite_improved_resolves():
    tool = TrainTool(policy_architecture="agalite_improved")
    assert isinstance(tool.policy_architecture, AGaLiTeImprovedConfig)


def test_policy_alias_transformer_resolves():
    tool = TrainTool(policy_architecture="transformer")
    assert isinstance(tool.policy_architecture, TransformerPolicyConfig)
