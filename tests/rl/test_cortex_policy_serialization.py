
import metta.agent.policies.cortex
import metta.rl.policy_artifact


def test_cortex_architecture_to_string_round_trip() -> None:
    cfg = metta.agent.policies.cortex.CortexBaseConfig()

    # Should not raise (previously failed due to live nn.Module serialization)
    spec = metta.rl.policy_artifact.policy_architecture_to_string(cfg)
    assert isinstance(spec, str) and len(spec) > 0
    assert "CortexBaseConfig" in spec

    # Round-trip back to a config
    reconstructed = metta.rl.policy_artifact.policy_architecture_from_string(spec)
    assert isinstance(reconstructed, metta.agent.policies.cortex.CortexBaseConfig)

    # The JSON views should match (stack field is excluded; stack_cfg is JSON-safe)
    assert reconstructed.model_dump(mode="json") == cfg.model_dump(mode="json")
