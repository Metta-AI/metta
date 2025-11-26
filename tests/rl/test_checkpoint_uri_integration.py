"""CheckpointManager URI integration tests aligned with the new training stack."""

import tempfile
from pathlib import Path

import pytest
import torch.nn as nn
from pydantic import Field

from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.mpt_artifact import load_mpt, save_mpt
from mettagrid.util.file import ParsedURI
from mettagrid.util.url_schemes import checkpoint_filename, key_and_version, resolve_uri


def create_checkpoint(tmp_path: Path, filename: str, architecture, state_dict) -> Path:
    checkpoint_path = tmp_path / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_mpt(checkpoint_path, architecture=architecture, state_dict=state_dict)
    return checkpoint_path


class _MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class _MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=_MockActionComponentConfig)

    def make_policy(self, policy_env_info):  # pragma: no cover - tests use provided agent
        return MockAgent()


@pytest.fixture
def mock_policy() -> MockAgent:
    return MockAgent()


@pytest.fixture
def mock_policy_architecture() -> _MockAgentPolicyArchitecture:
    return _MockAgentPolicyArchitecture()


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SystemConfig(data_dir=Path(tmpdir), local_only=True)


class TestFileURIs:
    def test_load_single_file_uri(self, tmp_path: Path, mock_policy, mock_policy_architecture):
        ckpt = create_checkpoint(
            tmp_path, checkpoint_filename("run", 5), mock_policy_architecture, mock_policy.state_dict()
        )
        artifact = load_mpt(str(ckpt))
        assert artifact.architecture is not None
        assert artifact.state_dict is not None

    def test_load_from_directory_latest(self, tmp_path: Path, mock_policy, mock_policy_architecture):
        ckpt_dir = tmp_path / "run" / "checkpoints"
        create_checkpoint(ckpt_dir, checkpoint_filename("run", 3), mock_policy_architecture, mock_policy.state_dict())
        latest = create_checkpoint(
            ckpt_dir, checkpoint_filename("run", 7), mock_policy_architecture, mock_policy.state_dict()
        )

        uri = f"file://{ckpt_dir}:latest"
        normalized = resolve_uri(uri)
        assert ":v7.mpt" in normalized
        assert latest.exists()

    def test_invalid_file_uri(self):
        with pytest.raises(FileNotFoundError):
            load_mpt("file:///does/not/exist.mpt")


class TestS3URIs:
    def test_key_and_version_parsing(self):
        key, version = key_and_version("s3://bucket/foo/checkpoints/foo:v9.mpt")
        assert key == "foo"
        assert version == 9


class TestCheckpointManagerOperations:
    def test_save_checkpoint_exists(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        ckpt_path = manager.checkpoint_dir / checkpoint_filename("demo", 1)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        save_mpt(ckpt_path, architecture=mock_policy_architecture, state_dict=mock_policy.state_dict())
        assert ckpt_path.exists()

    def test_latest_checkpoint_sorted(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        for epoch in [1, 3]:
            ckpt_path = manager.checkpoint_dir / checkpoint_filename("demo", epoch)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_mpt(ckpt_path, architecture=mock_policy_architecture, state_dict=mock_policy.state_dict())

        uri = manager.get_latest_checkpoint()
        assert uri is not None
        assert uri.endswith(":v3.mpt")

    def test_resolve_uri(self, tmp_path: Path, mock_policy_architecture, mock_policy):
        path = tmp_path / "model.mpt"
        create_checkpoint(tmp_path, path.name, mock_policy_architecture, mock_policy.state_dict())
        normalized = resolve_uri(str(path))
        assert normalized == f"file://{path}"


class TestMettaURIs:
    def test_parsed_uri_parses_metta_scheme(self):
        parsed = ParsedURI.parse("metta://policy/acee831a-f409-4345-9c44-79b34af17c3e")
        assert parsed.scheme == "metta"
        assert parsed.path == "policy/acee831a-f409-4345-9c44-79b34af17c3e"

    def test_resolve_metta_uri_invalid_format(self):
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="Unsupported metta:// URI format"):
            resolver.resolve("metta://invalid")

    def test_resolve_metta_uri_invalid_uuid(self):
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="Invalid policy version ID"):
            resolver.resolve("metta://policy/not-a-uuid")

    def test_resolve_metta_uri_requires_stats_server(self, monkeypatch):
        monkeypatch.setattr("metta.tools.utils.auto_config.auto_stats_server_uri", lambda: None)
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="stats server not configured"):
            resolver.resolve("metta://policy/acee831a-f409-4345-9c44-79b34af17c3e")
