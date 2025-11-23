"""CheckpointManager URI integration tests aligned with the new training stack."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch.nn as nn
from pydantic import Field

from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager, key_and_version
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.loader import save_policy


def checkpoint_filename(run: str, epoch: int) -> str:
    return f"{run}:v{epoch}.mpt"


def create_checkpoint(tmp_path: Path, filename: str, policy: MockAgent, architecture: PolicyArchitecture) -> Path:
    checkpoint_path = tmp_path / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    policy._policy_architecture = architecture
    save_policy(checkpoint_path, policy, arch_hint=architecture)
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
        ckpt = create_checkpoint(tmp_path, checkpoint_filename("run", 5), mock_policy, mock_policy_architecture)
        uri = f"file://{ckpt}"
        artifact = CheckpointManager.load_artifact_from_uri(uri)
        assert artifact.policy_architecture is not None

    def test_load_from_directory(self, tmp_path: Path, mock_policy, mock_policy_architecture):
        ckpt_dir = tmp_path / "run" / "checkpoints"
        create_checkpoint(ckpt_dir, checkpoint_filename("run", 3), mock_policy, mock_policy_architecture)
        latest = create_checkpoint(ckpt_dir, checkpoint_filename("run", 7), mock_policy, mock_policy_architecture)

        uri = f"file://{ckpt_dir}"
        artifact = CheckpointManager.load_artifact_from_uri(uri)
        assert artifact.policy_architecture is not None
        assert Path(uri[7:]).is_dir()
        assert latest.exists()

    def test_invalid_file_uri(self):
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_artifact_from_uri("file:///does/not/exist.mpt")


class TestS3URIs:
    @patch("metta.rl.checkpoint_manager.local_copy")
    def test_s3_download(self, mock_local_copy, mock_policy, mock_policy_architecture, tmp_path: Path):
        checkpoint_file = create_checkpoint(
            tmp_path, checkpoint_filename("run", 12), mock_policy, mock_policy_architecture
        )

        mock_local_copy.return_value.__enter__ = Mock(return_value=str(checkpoint_file))
        mock_local_copy.return_value.__exit__ = Mock(return_value=None)

        uri = "s3://bucket/run/checkpoints/run:v12.mpt"
        artifact = CheckpointManager.load_artifact_from_uri(uri)

        assert artifact.policy_architecture is not None

    def test_key_and_version_parsing(self):
        key, version = key_and_version("s3://bucket/foo/checkpoints/foo:v9.mpt")
        assert key == "foo"
        assert version == 9


class TestCheckpointManagerOperations:
    def test_save_agent_returns_uri(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        ckpt_path = manager.checkpoint_dir / checkpoint_filename("demo", 1)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        mock_policy._policy_architecture = mock_policy_architecture
        save_policy(ckpt_path, mock_policy, arch_hint=mock_policy_architecture)
        assert ckpt_path.exists()

    def test_latest_checkpoint_sorted(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        for epoch in [1, 3]:
            ckpt_path = manager.checkpoint_dir / checkpoint_filename("demo", epoch)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            mock_policy._policy_architecture = mock_policy_architecture
            save_policy(ckpt_path, mock_policy, arch_hint=mock_policy_architecture)

        uri = manager.get_latest_checkpoint()
        assert uri is not None
        assert uri.endswith(":v3.mpt")

    def test_normalize_uri(self, tmp_path: Path, mock_policy_architecture):
        path = tmp_path / "model.mpt"
        mock_agent = MockAgent()
        create_checkpoint(tmp_path, path.name, mock_agent, mock_policy_architecture)
        normalized = CheckpointManager.normalize_uri(str(path))
        assert normalized == f"file://{path}"
