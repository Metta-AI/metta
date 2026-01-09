"""Test that mettagrid.policy package exports are correct and all submodules are importable."""


def test_policy_package_exists():
    """Test that mettagrid.policy can be imported as a package."""
    import mettagrid.policy

    assert hasattr(mettagrid.policy, "__path__"), "mettagrid.policy should be a package"


def test_core_policy_classes_importable():
    """Test that core policy classes can be imported from mettagrid.policy submodules."""
    # These are the most commonly used imports across the codebase
    from mettagrid.policy.policy import (
        AgentPolicy,
        MultiAgentPolicy,
        NimMultiAgentPolicy,
        PolicySpec,
        StatefulAgentPolicy,
        StatefulPolicyImpl,
    )

    assert AgentPolicy is not None
    assert MultiAgentPolicy is not None
    assert NimMultiAgentPolicy is not None
    assert PolicySpec is not None
    assert StatefulAgentPolicy is not None
    assert StatefulPolicyImpl is not None


def test_policy_env_interface_importable():
    """Test that PolicyEnvInterface can be imported."""
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface

    assert PolicyEnvInterface is not None


def test_policy_loader_utilities_importable():
    """Test that policy loader utilities can be imported."""
    from mettagrid.policy.loader import (
        discover_and_register_policies,
        initialize_or_load_policy,
        resolve_policy_class_path,
    )

    assert discover_and_register_policies is not None
    assert initialize_or_load_policy is not None
    assert resolve_policy_class_path is not None


def test_policy_registry_importable():
    """Test that policy registry can be imported."""
    from mettagrid.policy.policy_registry import get_policy_registry

    assert get_policy_registry is not None
    # Should return a dict
    assert isinstance(get_policy_registry(), dict)


def test_submission_utilities_importable():
    """Test that submission utilities can be imported."""
    from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec

    assert POLICY_SPEC_FILENAME == "policy_spec.json"
    assert SubmissionPolicySpec is not None


def test_prepare_policy_spec_importable():
    """Test that prepare_policy_spec utilities can be imported."""
    from mettagrid.policy.prepare_policy_spec import download_policy_spec_from_s3_as_zip

    assert download_policy_spec_from_s3_as_zip is not None


def test_policy_implementations_importable():
    """Test that common policy implementations can be imported."""
    from mettagrid.policy.lstm import LSTMPolicy
    from mettagrid.policy.noop import NoopPolicy
    from mettagrid.policy.random_agent import RandomMultiAgentPolicy

    assert RandomMultiAgentPolicy is not None
    assert LSTMPolicy is not None
    assert NoopPolicy is not None


def test_token_encoder_importable():
    """Test that token encoder utilities can be imported."""
    from mettagrid.policy.token_encoder import coordinates

    assert coordinates is not None


def test_all_exports_from_init():
    """Test that __all__ exports from __init__.py are importable."""
    import mettagrid.policy

    # All items in __all__ should be importable from the package
    if hasattr(mettagrid.policy, "__all__"):
        for name in mettagrid.policy.__all__:
            assert hasattr(mettagrid.policy, name), f"{name} in __all__ but not available in mettagrid.policy"
