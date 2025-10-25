from metta.agent.policies.fast import FastConfig
from metta.rl.training import TrainingEnvironmentConfig
from metta.tools.train import TrainTool


def minimal_training_env() -> dict:
    return TrainingEnvironmentConfig().model_dump()


def test_policy_alias_fast_resolves() -> None:
    tool = TrainTool.model_validate({"policy_architecture": "fast", "training_env": minimal_training_env()})
    assert isinstance(tool.policy_architecture, FastConfig)
