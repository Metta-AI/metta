import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field

import mettagrid.builder.envs as eb
from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.common.util.uri import ParsedURI
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_artifact import load_policy_artifact, policy_spec_from_uri, save_policy_artifact
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy, save_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:  # pragma: no cover - simple stub
        return nn.Identity()


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, policy_env_info):  # pragma: no cover - tests use provided agent
        return MockAgent()


class ParameterizedMockAgent(MockAgent):
    def __init__(self, policy_env_info: PolicyEnvInterface | None = None):
        super().__init__(policy_env_info)
        self.dummy_weight = torch.nn.Parameter(torch.tensor([0.0]))


class ParameterizedMockArchitecture(PolicyArchitecture):
    class_path: str = "tests.rl.test_checkpoint_manager.ParameterizedMockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> ParameterizedMockAgent:  # pragma: no cover
        # simple test helper
        return ParameterizedMockAgent(policy_env_info)


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SystemConfig(data_dir=Path(tmpdir), local_only=True)


@pytest.fixture
def checkpoint_manager(test_system_cfg):
    return CheckpointManager(run="test_run", system_cfg=test_system_cfg)


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_policy_architecture():
    return MockAgentPolicyArchitecture()


class TestBasicSaveLoad:
    def test_load_from_uri_with_latest(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Test loading policy with :latest selector."""
        for epoch in [1, 7, 3]:
            save_policy_artifact(
                checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v{epoch}.mpt",
                policy_architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}:latest"
        artifact = load_policy_artifact(latest_uri)

        assert artifact.state_dict is not None
        metadata = CheckpointManager.get_policy_metadata(latest_uri)
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 7

    def test_save_and_load_agent(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        checkpoint_dir = checkpoint_manager.checkpoint_dir
        expected_filename = "test_run:v5.mpt"
        agent_file = checkpoint_dir / expected_filename

        assert agent_file.exists()

        metadata = CheckpointManager.get_policy_metadata(agent_file.as_uri())
        assert metadata["run_name"] == "test_run"
        assert metadata["epoch"] == 5

        artifact = load_policy_artifact(agent_file.as_uri())
        assert artifact.state_dict is not None

    def test_remote_prefix_upload(self, test_system_cfg, mock_agent, mock_policy_architecture):
        test_system_cfg.local_only = False
        test_system_cfg.remote_prefix = "s3://bucket/checkpoints"
        manager = CheckpointManager(run="test_run", system_cfg=test_system_cfg)

        expected_filename = "test_run:v3.mpt"
        local_path = manager.checkpoint_dir / expected_filename

        save_policy_artifact(
            local_path,
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        # Simulate remote prefix using a local directory
        remote_dir = manager.checkpoint_dir / "remote"
        remote_dir.mkdir(parents=True, exist_ok=True)

        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy_spec = policy_spec_from_uri(f"file://{local_path}")
        policy = initialize_or_load_policy(env_info, policy_spec)
        remote_path = remote_dir / expected_filename
        remote_uri = save_policy(remote_path, policy, arch_hint=mock_policy_architecture)

        assert remote_path.exists()
        # URI may percent-encode ':'; compare normalized parts
        assert Path(ParsedURI.parse(remote_uri).local_path) == remote_path
        assert remote_path.exists()

    def test_multiple_epoch_saves_and_selection(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        epochs = [1, 5, 10]

        for epoch in epochs:
            save_policy_artifact(
                checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v{epoch}.mpt",
                policy_architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint.endswith(":v10.mpt")
        artifact = load_policy_artifact(latest_checkpoint)
        assert artifact.state_dict is not None

    def test_trainer_state_save_load(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        stopwatch_state = {"elapsed_time": 123.45}
        checkpoint_manager.save_trainer_state(mock_optimizer, epoch=5, agent_step=1000, stopwatch_state=stopwatch_state)

        loaded_trainer_state = checkpoint_manager.load_trainer_state()
        assert loaded_trainer_state is not None
        assert loaded_trainer_state["epoch"] == 5
        assert loaded_trainer_state["agent_step"] == 1000
        assert loaded_trainer_state["stopwatch_state"]["elapsed_time"] == 123.45
        assert loaded_trainer_state.get("loss_states", {}) == {}
        assert "optimizer_state" in loaded_trainer_state

    def test_checkpoint_existence(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None

        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v1.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

    def test_policy_spec_from_uri_initializes_checkpoint(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v1.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = policy_spec_from_uri(latest)
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)
        assert policy.agent_policy(0) is not None

    def test_policy_spec_respects_strict_flag(self, tmp_path: Path):
        """Ensure strict=False in spec suppresses load errors."""
        arch = ParameterizedMockArchitecture()
        bad_state = {"unexpected": torch.tensor([1.0])}
        checkpoint_path = tmp_path / "bad.mpt"
        save_policy_artifact(
            checkpoint_path,
            policy_architecture=arch,
            state_dict=bad_state,
        )

        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        spec = policy_spec_from_uri(checkpoint_path.as_uri(), strict=False)
        policy = initialize_or_load_policy(env_info, spec)
        assert isinstance(policy, ParameterizedMockAgent)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_policy_spec_moves_policy_to_device(self, tmp_path: Path):
        """Ensure device hint in spec moves loaded policy."""
        arch = ParameterizedMockArchitecture()
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = arch.make_policy(env_info)

        checkpoint_path = tmp_path / "device.mpt"
        save_policy_artifact(
            checkpoint_path,
            policy_architecture=arch,
            state_dict=policy.state_dict(),
        )

        spec = policy_spec_from_uri(checkpoint_path.as_uri(), device=torch.device("cuda:0"))
        loaded = initialize_or_load_policy(env_info, spec)
        assert isinstance(loaded, ParameterizedMockAgent)
        assert next(loaded.parameters()).device.type == "cuda"

    def test_checkpoint_policy_save_policy_round_trip(
        self,
        checkpoint_manager,
        mock_agent,
        mock_policy_architecture,
    ):
        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v5.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        spec = policy_spec_from_uri(latest)
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = initialize_or_load_policy(env_info, spec)

        save_path = checkpoint_manager.checkpoint_dir / "resaved.mpt"
        saved_uri = save_policy(save_path, policy, arch_hint=mock_policy_architecture)

        assert save_path.exists()
        reload_spec = policy_spec_from_uri(saved_uri, device="cpu")
        reloaded = initialize_or_load_policy(env_info, reload_spec)
        assert reloaded is not None

    def test_policy_spec_from_uri_embedded_metadata(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        save_policy_artifact(
            checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v2.mpt",
            policy_architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

        spec = policy_spec_from_uri(latest, display_name="custom", device="cpu")
        assert isinstance(spec, PolicySpec)
        assert spec.class_path == mock_policy_architecture.class_path
        assert spec.init_kwargs is not None
        assert spec.init_kwargs["checkpoint_uri"] == CheckpointManager.normalize_uri(latest)
        assert spec.init_kwargs["display_name"] == "custom"
        assert spec.init_kwargs["device"] == "cpu"
        assert spec.init_kwargs.get("policy_architecture") is not None

    def test_weights_only_checkpoint_loads_with_class_path(self, tmp_path: Path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = ParameterizedMockAgent(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(11.0)

        save_path = tmp_path / "weights_only.pt"
        torch.save(policy.state_dict(), save_path)

        spec = policy_spec_from_uri(
            save_path.as_uri(),
            class_path="tests.rl.test_checkpoint_manager.ParameterizedMockAgent",
        )

        restored = initialize_or_load_policy(env_info, spec)
        assert isinstance(restored, ParameterizedMockAgent)
        assert torch.allclose(restored.dummy_weight, torch.tensor([11.0]))

    def test_weights_only_checkpoint_loads_with_policy_architecture(self, tmp_path: Path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = ParameterizedMockAgent(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(13.0)

        save_path = tmp_path / "weights_only.pt"
        torch.save(policy.state_dict(), save_path)

        arch = ParameterizedMockArchitecture()
        spec = policy_spec_from_uri(
            save_path.as_uri(),
            policy_architecture=arch,
        )

        restored = initialize_or_load_policy(env_info, spec)
        assert isinstance(restored, ParameterizedMockAgent)
        assert torch.allclose(restored.dummy_weight, torch.tensor([13.0]))

    def test_loader_uses_checkpoint_uri_when_data_path_missing(self, checkpoint_manager):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        arch = ParameterizedMockArchitecture()
        policy = arch.make_policy(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(5.0)

        ckpt_path = checkpoint_manager.checkpoint_dir / f"{checkpoint_manager.run_name}:v_uri.mpt"
        save_policy_artifact(
            ckpt_path,
            policy_architecture=arch,
            state_dict=policy.state_dict(),
        )

        spec = PolicySpec(
            class_path=arch.class_path,
            data_path=None,
            init_kwargs={
                "checkpoint_uri": ckpt_path.as_uri(),
                "policy_architecture": arch,
            },
        )

        restored = initialize_or_load_policy(env_info, spec)
        assert torch.allclose(restored.dummy_weight, torch.tensor([5.0]))

    def test_loader_supports_serialized_policy_objects(self, tmp_path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = ParameterizedMockAgent(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(3.0)

        save_path = tmp_path / "policy.pt"
        torch.save(policy, save_path)

        spec = PolicySpec(
            class_path="tests.rl.test_checkpoint_manager.ParameterizedMockAgent",
            data_path=None,
            init_kwargs={"checkpoint_uri": save_path.as_uri()},
        )

        restored = initialize_or_load_policy(env_info, spec)
        assert torch.allclose(restored.dummy_weight, torch.tensor([3.0]))

    def test_loader_prefers_embedded_policy_over_arch_hint(self, tmp_path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = ParameterizedMockAgent(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(9.0)

        save_path = tmp_path / "embedded.pt"
        torch.save(policy, save_path)

        spec = policy_spec_from_uri(save_path.as_uri())
        arch_hint = ParameterizedMockArchitecture()

        restored = initialize_or_load_policy(env_info, spec, arch_hint=arch_hint)
        assert isinstance(restored, ParameterizedMockAgent)
        assert torch.allclose(restored.dummy_weight, torch.tensor([9.0]))

    def test_policy_spec_from_directory_uri_does_not_set_data_path(self, tmp_path: Path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        arch = ParameterizedMockArchitecture()
        policy = arch.make_policy(env_info)

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        save_policy_artifact(
            ckpt_dir / "run:v1.mpt",
            policy_architecture=arch,
            state_dict=policy.state_dict(),
        )

        spec = policy_spec_from_uri(ckpt_dir.as_uri())
        assert spec.data_path is None

        restored = initialize_or_load_policy(env_info, spec)
        assert isinstance(restored, ParameterizedMockAgent)

    def test_save_policy_to_s3_uses_temp_file(self, tmp_path: Path, monkeypatch, mock_policy_architecture):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        policy = mock_policy_architecture.make_policy(env_info)

        uploads: dict[str, str] = {}

        def fake_write_file(path: str, local_file: str, **_: str) -> None:
            uploads["path"] = path
            uploads["local"] = local_file

        def fake_guess_data_dir() -> Path:
            return tmp_path / "data"

        monkeypatch.setattr("metta.common.util.file.write_file", fake_write_file)
        monkeypatch.setattr("metta.rl.system_config.guess_data_dir", fake_guess_data_dir)

        dest = "s3://bucket/run/foo.mpt"
        result = save_policy(dest, policy, arch_hint=mock_policy_architecture)

        assert result == dest
        assert uploads["path"] == dest
        assert Path(uploads["local"]).parent == fake_guess_data_dir() / ".tmp"
        assert not Path(uploads["local"]).exists()
        assert not (tmp_path / "s3:").exists()

    def test_loader_falls_back_to_checkpoint_uri_when_data_path_missing(self, tmp_path: Path):
        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        arch = ParameterizedMockArchitecture()
        policy = arch.make_policy(env_info)
        with torch.no_grad():
            policy.dummy_weight.fill_(17.0)

        ckpt_path = tmp_path / "fallback.mpt"
        save_policy_artifact(
            ckpt_path,
            policy_architecture=arch,
            state_dict=policy.state_dict(),
        )

        spec = PolicySpec(
            class_path=arch.class_path,
            data_path=str(tmp_path / "missing.mpt"),
            init_kwargs={
                "checkpoint_uri": ckpt_path.as_uri(),
                "policy_architecture": arch,
            },
        )

        restored = initialize_or_load_policy(env_info, spec)
        assert torch.allclose(restored.dummy_weight, torch.tensor([17.0]))


class TestErrorHandling:
    def test_load_from_empty_directory(self, checkpoint_manager):
        result = checkpoint_manager.load_trainer_state()
        assert result is None

        checkpoints = checkpoint_manager.get_latest_checkpoint()
        assert checkpoints is None

    def test_invalid_run_name(self, test_system_cfg):
        invalid_names = ["", "name with spaces", "name/with/slash", "name*with*asterisk"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=invalid_name, system_cfg=test_system_cfg)
