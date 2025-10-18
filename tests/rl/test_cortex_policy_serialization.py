from __future__ import annotations

from metta.agent.policies.cortex import CortexBaseConfig
from metta.rl.policy_artifact import policy_architecture_from_string, policy_architecture_to_string


def test_cortex_architecture_to_string_round_trip() -> None:
    cfg = CortexBaseConfig()

    # Should not raise (previously failed due to live nn.Module serialization)
    spec = policy_architecture_to_string(cfg)
    assert isinstance(spec, str) and len(spec) > 0
    assert "CortexBaseConfig" in spec

    # Round-trip back to a config
    reconstructed = policy_architecture_from_string(spec)
    assert isinstance(reconstructed, CortexBaseConfig)

    # The JSON views should match (stack field is excluded; stack_cfg is JSON-safe)
    assert reconstructed.model_dump(mode="json") == cfg.model_dump(mode="json")
