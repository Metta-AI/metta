import pytest

from metta.tools.utils import auto_config


def _reset_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPLAY_DIR", raising=False)
    monkeypatch.delenv("POLICY_REMOTE_PREFIX", raising=False)
    monkeypatch.setattr(
        auto_config,
        "supported_aws_env_overrides",
        auto_config.SupportedAwsEnvOverrides(),
        raising=False,
    )


def test_auto_policy_storage_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_overrides(monkeypatch)
    monkeypatch.setenv("POLICY_REMOTE_PREFIX", "s3://custom-bucket/policies")
    monkeypatch.setattr(
        auto_config,
        "supported_aws_env_overrides",
        auto_config.SupportedAwsEnvOverrides(),
        raising=False,
    )

    decision = auto_config.auto_policy_storage_decision("demo-run")

    assert decision.using_remote is True
    assert decision.reason == "env_override"
    assert decision.remote_prefix == "s3://custom-bucket/policies/demo-run"


def test_auto_policy_storage_softmax_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_overrides(monkeypatch)

    class DummyAWSSetup:
        def is_enabled(self) -> bool:
            return True

        def to_config_settings(self) -> dict[str, str | bool]:
            return {"policy_remote_prefix": "s3://softmax-public/policies"}

        def check_connected_as(self) -> str:
            return auto_config.METTA_AWS_ACCOUNT_ID

    monkeypatch.setattr(auto_config, "AWSSetup", lambda: DummyAWSSetup(), raising=False)

    decision = auto_config.auto_policy_storage_decision("softmax-run")

    assert decision.using_remote is True
    assert decision.reason == "softmax_connected"
    assert decision.remote_prefix == "s3://softmax-public/policies/softmax-run"


def test_auto_policy_storage_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_overrides(monkeypatch)

    class DummyAWSSetup:
        def is_enabled(self) -> bool:
            return True

        def to_config_settings(self) -> dict[str, str | bool]:
            return {"policy_remote_prefix": "s3://softmax-public/policies"}

        def check_connected_as(self) -> str | None:
            return None

    monkeypatch.setattr(auto_config, "AWSSetup", lambda: DummyAWSSetup(), raising=False)

    decision = auto_config.auto_policy_storage_decision("offline-run")

    assert decision.using_remote is False
    assert decision.reason == "not_connected"
    assert decision.remote_prefix is None


def test_auto_policy_storage_not_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_overrides(monkeypatch)

    class DummyAWSSetup:
        def is_enabled(self) -> bool:
            return False

        def to_config_settings(self) -> dict[str, str | bool]:
            return {}

        def check_connected_as(self) -> str | None:
            return None

    monkeypatch.setattr(auto_config, "AWSSetup", lambda: DummyAWSSetup(), raising=False)

    decision = auto_config.auto_policy_storage_decision("local-run")

    assert decision.using_remote is False
    assert decision.reason == "aws_not_enabled"
    assert decision.remote_prefix is None


def test_auto_policy_storage_missing_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_overrides(monkeypatch)

    class DummyAWSSetup:
        def is_enabled(self) -> bool:
            return True

        def to_config_settings(self) -> dict[str, str | bool]:
            return {"replay_dir": "s3://softmax-public/replays/"}

        def check_connected_as(self) -> str:
            return auto_config.METTA_AWS_ACCOUNT_ID

    monkeypatch.setattr(auto_config, "AWSSetup", lambda: DummyAWSSetup(), raising=False)

    decision = auto_config.auto_policy_storage_decision("mismatch-run")

    assert decision.using_remote is False
    assert decision.reason == "no_base_prefix"
    assert decision.remote_prefix is None
