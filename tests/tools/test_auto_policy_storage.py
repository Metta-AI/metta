from dataclasses import dataclass
from functools import partial
from typing import Callable

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


def _setup_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_REMOTE_PREFIX", "s3://custom-bucket/policies")
    monkeypatch.setattr(
        auto_config,
        "supported_aws_env_overrides",
        auto_config.SupportedAwsEnvOverrides(),
        raising=False,
    )


def _patch_aws_setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    prefix: str | None,
    connected_as: str | None,
    extra_settings: dict[str, str | bool] | None = None,
) -> None:
    class DummyAWSSetup:
        def is_enabled(self) -> bool:
            return enabled

        def to_config_settings(self) -> dict[str, str | bool]:
            settings = dict(extra_settings or {})
            if prefix is not None:
                settings["policy_remote_prefix"] = prefix
            return settings

        def check_connected_as(self) -> str | None:
            return connected_as

    monkeypatch.setattr(auto_config, "AWSSetup", lambda: DummyAWSSetup(), raising=False)


@dataclass(frozen=True)
class Scenario:
    id: str
    run_name: str
    setup: Callable[[pytest.MonkeyPatch], None]
    expected_using_remote: bool
    expected_reason: str
    expected_base_prefix: str | None
    expected_remote_prefix: str | None


_SCENARIOS: list[Scenario] = [
    Scenario(
        id="env_override",
        run_name="demo-run",
        setup=_setup_env_override,
        expected_using_remote=True,
        expected_reason="env_override",
        expected_base_prefix="s3://custom-bucket/policies",
        expected_remote_prefix="s3://custom-bucket/policies/demo-run",
    ),
    Scenario(
        id="softmax_connected",
        run_name="softmax-run",
        setup=partial(
            _patch_aws_setup,
            enabled=True,
            prefix="s3://softmax-public/policies",
            connected_as=auto_config.METTA_AWS_ACCOUNT_ID,
            extra_settings=None,
        ),
        expected_using_remote=True,
        expected_reason="softmax_connected",
        expected_base_prefix="s3://softmax-public/policies",
        expected_remote_prefix="s3://softmax-public/policies/softmax-run",
    ),
    Scenario(
        id="not_connected",
        run_name="offline-run",
        setup=partial(
            _patch_aws_setup,
            enabled=True,
            prefix="s3://softmax-public/policies",
            connected_as=None,
            extra_settings=None,
        ),
        expected_using_remote=False,
        expected_reason="not_connected",
        expected_base_prefix="s3://softmax-public/policies",
        expected_remote_prefix=None,
    ),
    Scenario(
        id="aws_not_enabled",
        run_name="local-run",
        setup=partial(
            _patch_aws_setup,
            enabled=False,
            prefix=None,
            connected_as=None,
            extra_settings=None,
        ),
        expected_using_remote=False,
        expected_reason="aws_not_enabled",
        expected_base_prefix=None,
        expected_remote_prefix=None,
    ),
    Scenario(
        id="missing_prefix",
        run_name="mismatch-run",
        setup=partial(
            _patch_aws_setup,
            enabled=True,
            prefix=None,
            connected_as=auto_config.METTA_AWS_ACCOUNT_ID,
            extra_settings={"replay_dir": "s3://softmax-public/replays/"},
        ),
        expected_using_remote=False,
        expected_reason="no_base_prefix",
        expected_base_prefix=None,
        expected_remote_prefix=None,
    ),
]


@pytest.mark.skip(reason="Test temporarily disabled")
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[scenario.id for scenario in _SCENARIOS])
def test_auto_policy_storage_decision(monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> None:
    _reset_overrides(monkeypatch)
    scenario.setup(monkeypatch)

    decision = auto_config.auto_policy_storage_decision(scenario.run_name)

    assert decision.using_remote is scenario.expected_using_remote
    assert decision.reason == scenario.expected_reason
    assert decision.base_prefix == scenario.expected_base_prefix
    assert decision.remote_prefix == scenario.expected_remote_prefix
