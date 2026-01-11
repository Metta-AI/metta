from __future__ import annotations

from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy_architecture import PolicyArchitecture


def test_cortex_architecture_to_spec_round_trip() -> None:
    cfg = CortexBaseConfig()

    spec = cfg.to_spec()
    assert isinstance(spec, str) and len(spec) > 0
    assert "CortexBaseConfig" in spec

    reconstructed = PolicyArchitecture.from_spec(spec)
    assert isinstance(reconstructed, CortexBaseConfig)

    assert reconstructed.model_dump(mode="json") == cfg.model_dump(mode="json")
