"""CheckpointManager URI integration tests aligned with the new training stack."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from metta.rl.checkpoint_manager import CheckpointManager, key_and_version
from metta.rl.system_config import SystemConfig


def checkpoint_filename(run: str, epoch: int) -> str:
    return f"{run}:v{epoch}.pt"


def create_checkpoint(tmp_path: Path, filename: str, payload) -> Path:
    checkpoint_path = tmp_path / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_policy():
    return torch.nn.Linear(4, 2)


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SystemConfig(data_dir=Path(tmpdir), local_only=True)


class TestFileURIs:
    def test_load_single_file_uri(self, tmp_path: Path, mock_policy):
        ckpt = create_checkpoint(tmp_path, checkpoint_filename("run", 5), mock_policy)
        uri = f"file://{ckpt}"
        loaded = CheckpointManager.load_from_uri(uri)
        assert isinstance(loaded, torch.nn.Module)

    def test_load_from_directory(self, tmp_path: Path, mock_policy):
        ckpt_dir = tmp_path / "run" / "checkpoints"
        create_checkpoint(ckpt_dir, checkpoint_filename("run", 3), mock_policy)
        latest = create_checkpoint(ckpt_dir, checkpoint_filename("run", 7), mock_policy)

        uri = f"file://{ckpt_dir}"
        loaded = CheckpointManager.load_from_uri(uri)
        assert isinstance(loaded, torch.nn.Module)
        assert Path(uri[7:]).is_dir()
        assert latest.exists()

    def test_invalid_file_uri(self):
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri("file:///does/not/exist.pt")


class TestS3URIs:
    @patch("metta.rl.checkpoint_manager.local_copy")
    def test_s3_download(self, mock_local_copy, mock_policy):
        mock_local_copy.return_value.__enter__ = Mock(return_value="/tmp/downloaded.pt")
        mock_local_copy.return_value.__exit__ = Mock(return_value=None)

        with patch("torch.load", return_value=mock_policy) as mocked_load:
            uri = "s3://bucket/run/checkpoints/run:v12.pt"
            loaded = CheckpointManager.load_from_uri(uri)

        mocked_load.assert_called_once()
        assert isinstance(loaded, torch.nn.Module)

    def test_key_and_version_parsing(self):
        key, version = key_and_version("s3://bucket/foo/checkpoints/foo:v9.pt")
        assert key == "foo"
        assert version == 9


class TestCheckpointManagerOperations:
    def test_save_agent_returns_uri(self, test_system_cfg, mock_policy):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        uri = manager.save_agent(mock_policy, epoch=1, metadata={})
        assert uri.startswith("file://")
        saved_path = Path(uri[7:])
        assert saved_path.exists()

    def test_select_checkpoints_sorted(self, test_system_cfg, mock_policy):
        manager = CheckpointManager(run="demo", system_cfg=test_system_cfg)
        manager.save_agent(mock_policy, epoch=1, metadata={})
        manager.save_agent(mock_policy, epoch=3, metadata={})
        uris = manager.select_checkpoints(strategy="latest", count=1)
        assert len(uris) == 1
        assert uris[0].endswith(":v3.pt")

    def test_normalize_uri(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        torch.save(torch.nn.Linear(1, 1), path)
        normalized = CheckpointManager.normalize_uri(str(path))
        assert normalized == f"file://{path}"
