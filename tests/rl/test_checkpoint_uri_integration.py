"""CheckpointManager URI integration tests aligned with the new training stack."""

import pathlib
import tempfile
import unittest.mock

import pydantic
import pytest
import torch.nn as nn

import metta.agent.components.component_config
import metta.agent.mocks
import metta.agent.policy
import metta.rl.checkpoint_manager
import metta.rl.policy_artifact
import metta.rl.system_config
import mettagrid.base_config


def checkpoint_filename(run: str, epoch: int) -> str:
    return f"{run}:v{epoch}.mpt"


def create_checkpoint(tmp_path: pathlib.Path, filename: str, policy: metta.agent.mocks.MockAgent) -> pathlib.Path:
    checkpoint_path = tmp_path / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metta.rl.policy_artifact.save_policy_artifact_pt(checkpoint_path, policy=policy)
    return checkpoint_path


class _MockActionComponentConfig(metta.agent.components.component_config.ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class _MockAgentPolicyArchitecture(metta.agent.policy.PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: mettagrid.base_config.Config = pydantic.Field(default_factory=_MockActionComponentConfig)

    def make_policy(self, policy_env_info):  # pragma: no cover - tests use provided agent
        return metta.agent.mocks.MockAgent()


@pytest.fixture
def mock_policy() -> metta.agent.mocks.MockAgent:
    return metta.agent.mocks.MockAgent()


@pytest.fixture
def mock_policy_architecture() -> _MockAgentPolicyArchitecture:
    return _MockAgentPolicyArchitecture()


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield metta.rl.system_config.SystemConfig(data_dir=pathlib.Path(tmpdir), local_only=True)


class TestFileURIs:
    def test_load_single_file_uri(self, tmp_path: pathlib.Path, mock_policy):
        ckpt = create_checkpoint(tmp_path, checkpoint_filename("run", 5), mock_policy)
        uri = f"file://{ckpt}"
        artifact = metta.rl.checkpoint_manager.CheckpointManager.load_artifact_from_uri(uri)
        assert artifact.policy is not None

    def test_load_from_directory(self, tmp_path: pathlib.Path, mock_policy):
        ckpt_dir = tmp_path / "run" / "checkpoints"
        create_checkpoint(ckpt_dir, checkpoint_filename("run", 3), mock_policy)
        latest = create_checkpoint(ckpt_dir, checkpoint_filename("run", 7), mock_policy)

        uri = f"file://{ckpt_dir}"
        artifact = metta.rl.checkpoint_manager.CheckpointManager.load_artifact_from_uri(uri)
        assert artifact.policy is not None
        assert pathlib.Path(uri[7:]).is_dir()
        assert latest.exists()

    def test_invalid_file_uri(self):
        with pytest.raises(FileNotFoundError):
            metta.rl.checkpoint_manager.CheckpointManager.load_artifact_from_uri("file:///does/not/exist.mpt")


class TestS3URIs:
    @unittest.mock.patch("metta.rl.checkpoint_manager.local_copy")
    def test_s3_download(self, mock_local_copy, mock_policy, tmp_path: pathlib.Path):
        checkpoint_file = create_checkpoint(tmp_path, checkpoint_filename("run", 12), mock_policy)

        mock_local_copy.return_value.__enter__ = unittest.mock.Mock(return_value=str(checkpoint_file))
        mock_local_copy.return_value.__exit__ = unittest.mock.Mock(return_value=None)

        uri = "s3://bucket/run/checkpoints/run:v12.mpt"
        artifact = metta.rl.checkpoint_manager.CheckpointManager.load_artifact_from_uri(uri)

        assert artifact.policy is not None

    def test_key_and_version_parsing(self):
        key, version = metta.rl.checkpoint_manager.key_and_version("s3://bucket/foo/checkpoints/foo:v9.mpt")
        assert key == "foo"
        assert version == 9


class TestCheckpointManagerOperations:
    def test_save_agent_returns_uri(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = metta.rl.checkpoint_manager.CheckpointManager(run="demo", system_cfg=test_system_cfg)
        uri = manager.save_agent(mock_policy, epoch=1, policy_architecture=mock_policy_architecture)
        assert uri.startswith("file://")
        saved_path = pathlib.Path(uri[7:])
        assert saved_path.exists()

    def test_latest_checkpoint_sorted(self, test_system_cfg, mock_policy, mock_policy_architecture):
        manager = metta.rl.checkpoint_manager.CheckpointManager(run="demo", system_cfg=test_system_cfg)
        manager.save_agent(mock_policy, epoch=1, policy_architecture=mock_policy_architecture)
        manager.save_agent(mock_policy, epoch=3, policy_architecture=mock_policy_architecture)

        uri = manager.get_latest_checkpoint()
        assert uri is not None
        assert uri.endswith(":v3.mpt")

    def test_normalize_uri(self, tmp_path: pathlib.Path):
        path = tmp_path / "model.mpt"
        create_checkpoint(tmp_path, path.name, metta.agent.mocks.MockAgent())
        normalized = metta.rl.checkpoint_manager.CheckpointManager.normalize_uri(str(path))
        assert normalized == f"file://{path}"
