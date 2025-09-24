from experiments.recipes import arena_basic_easy_shaped
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.smollm2 import SmolLM2Config


def test_arena_basic_default_policy_is_fast() -> None:
    cfg = arena_basic_easy_shaped.train()
    assert isinstance(cfg.policy_architecture, FastConfig)


def test_arena_basic_can_select_smollm2() -> None:
    cfg = arena_basic_easy_shaped.train(policy="smollm2", freeze_llm=False)
    assert isinstance(cfg.policy_architecture, SmolLM2Config)
    assert cfg.policy_architecture.freeze_llm is False
