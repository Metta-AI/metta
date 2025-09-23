"""CheckpointManager URI integration tests aligned with the new training stack."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager, key_and_version
from metta.rl.policy_serialization import load_policy_artifact, save_policy_artifact
from metta.rl.training.checkpointer import Checkpointer, CheckpointerConfig
from metta.rl.training.training_environment import EnvironmentMetaData


class MockPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    components: list = []
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="noop")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> MockAgent:  # type: ignore[override]
        return MockAgent()


@pytest.fixture
def env_metadata() -> EnvironmentMetaData:
    return EnvironmentMetaData(
        obs_width=1,
        obs_height=1,
        obs_features={},
        action_names=["noop"],
        max_action_args=[0],
        num_agents=1,
        observation_space=None,
        action_space=None,
        feature_normalizations={},
    )


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


class TestFileURIs:
    def test_load_single_file_uri(self, tmp_path: Path, mock_policy):
        ckpt = create_checkpoint(tmp_path, checkpoint_filename("run", 5), mock_policy)
        uri = f"file://{ckpt}"
        loaded = CheckpointManager.load_from_uri(uri)
        assert loaded.policy is not None
        assert isinstance(loaded.policy, torch.nn.Module)

    def test_load_from_directory(self, tmp_path: Path, mock_policy):
        ckpt_dir = tmp_path / "run" / "checkpoints"
        create_checkpoint(ckpt_dir, checkpoint_filename("run", 3), mock_policy)
        latest = create_checkpoint(ckpt_dir, checkpoint_filename("run", 7), mock_policy)

        uri = f"file://{ckpt_dir}"
        loaded = CheckpointManager.load_from_uri(uri)
        assert loaded.policy is not None
        assert isinstance(loaded.policy, torch.nn.Module)
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
        assert loaded.policy is not None
        assert isinstance(loaded.policy, torch.nn.Module)

    def test_key_and_version_parsing(self):
        key, version = key_and_version("s3://bucket/foo/checkpoints/foo:v9.pt")
        assert key == "foo"
        assert version == 9


class TestCheckpointManagerOperations:
    def test_save_agent_returns_uri(self, tmp_path: Path, mock_policy):
        manager = CheckpointManager(run="demo", run_dir=str(tmp_path))
        uri = manager.save_agent(mock_policy, epoch=1, training_metrics={})
        assert uri.startswith("file://")
        saved_path = Path(uri[7:])
        assert saved_path.exists()

    def test_select_checkpoints_sorted(self, tmp_path: Path, mock_policy):
        manager = CheckpointManager(run="demo", run_dir=str(tmp_path))
        manager.save_agent(mock_policy, epoch=1, training_metrics={})
        manager.save_agent(mock_policy, epoch=3, training_metrics={})
        uris = manager.select_checkpoints(strategy="latest", count=1)
        assert len(uris) == 1
        assert uris[0].endswith(":v3.pt")

    def test_normalize_uri(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        torch.save(torch.nn.Linear(1, 1), path)
        normalized = CheckpointManager.normalize_uri(str(path))
        assert normalized == f"file://{path}"


class TestPolicyArtifactSerialization:
    def test_round_trip(self, tmp_path: Path, env_metadata: EnvironmentMetaData) -> None:
        architecture = MockPolicyArchitecture()
        policy = MockAgent()
        base_path = tmp_path / "test" / "artifact"

        metrics = {"loss": 1.23, "accuracy": 0.99}
        weights_path, metrics_path = save_policy_artifact(
            base_path=base_path,
            policy=policy,
            policy_architecture=architecture,
            training_metrics=metrics,
        )

        manifest = {
            "class_path": f"{architecture.__class__.__module__}.{architecture.__class__.__qualname__}",
            "config": architecture.model_dump(mode="json"),
        }
        manifest_path = base_path.parent / "model_architecture.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        assert weights_path.exists()
        assert metrics_path.exists()

        artifact = load_policy_artifact(base_path)
        assert artifact.training_metrics == metrics

        reconstructed = artifact.instantiate(env_metadata)
        assert isinstance(reconstructed, MockAgent)


class TestCheckpointerManifest:
    class _DummyDistributed:
        def is_master(self) -> bool:
            return True

        def broadcast_from_master(self, value):
            return value

    def test_register_writes_manifest(self, tmp_path: Path) -> None:
        checkpoint_manager = CheckpointManager(run="demo", run_dir=str(tmp_path))
        distributed = self._DummyDistributed()
        architecture = MockPolicyArchitecture()
        checkpointer = Checkpointer(
            config=CheckpointerConfig(),
            checkpoint_manager=checkpoint_manager,
            distributed_helper=distributed,
            policy_architecture=architecture,
        )

        context = SimpleNamespace()
        checkpointer.register(context)

        manifest_path = checkpoint_manager.checkpoint_dir / "model_architecture.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["class_path"] == f"{architecture.__class__.__module__}.{architecture.__class__.__qualname__}"
        assert manifest["config"] == architecture.model_dump(mode="json")


class TestTrainingMetricsPersistence:
    def test_pt_checkpoint_writes_metrics(self, tmp_path: Path) -> None:
        manager = CheckpointManager(run="demo", run_dir=str(tmp_path), checkpoint_format="pt")
        metrics = {"loss": 0.42}
        uri = manager.save_agent(MockAgent(), epoch=5, training_metrics=metrics)

        checkpoint_path = Path(uri[7:])  # strip file://
        assert checkpoint_path.exists()
        assert not checkpoint_path.with_suffix(".metrics.json").exists()

    def test_safetensor_checkpoint_writes_metrics(self, tmp_path: Path, env_metadata: EnvironmentMetaData) -> None:
        manager = CheckpointManager(
            run="demo",
            run_dir=str(tmp_path),
            checkpoint_format="safetensors",
        )
        architecture = MockPolicyArchitecture()
        metrics = {"loss": 0.17, "accuracy": 0.88}

        (manager.checkpoint_dir / "model_architecture.json").parent.mkdir(parents=True, exist_ok=True)
        (manager.checkpoint_dir / "model_architecture.json").write_text(
            json.dumps(
                {
                    "class_path": f"{architecture.__class__.__module__}.{architecture.__class__.__qualname__}",
                    "config": architecture.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        uri = manager.save_agent(
            MockAgent(),
            epoch=2,
            training_metrics=metrics,
            policy_architecture=architecture,
        )

        checkpoint_path = Path(uri[7:])
        assert checkpoint_path.exists()
        metrics_path = checkpoint_path.with_suffix(".metrics.json")
        assert metrics_path.exists()
        persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert persisted == metrics
