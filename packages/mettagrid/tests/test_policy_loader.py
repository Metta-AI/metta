from mettagrid.policy.loader import policy_spec_from_string
from mettagrid.policy.policy import PolicySpec


def test_policy_spec_from_string_accepts_class_path() -> None:
    spec = policy_spec_from_string("cogames.policy.scripted_agent.baseline_agent.BaselinePolicy")

    assert isinstance(spec, PolicySpec)
    assert spec.class_path == "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy"
    assert spec.data_path is None


def test_policy_spec_from_string_handles_local_checkpoint_paths(tmp_path) -> None:
    checkpoint = tmp_path / "model.mpt"
    checkpoint.write_bytes(b"")

    spec = policy_spec_from_string(str(checkpoint))

    assert spec.class_path == "mettagrid.policy.mpt_policy.MptPolicy"
    assert spec.init_kwargs
    assert spec.init_kwargs.get("checkpoint_uri", "").endswith("model.mpt")
