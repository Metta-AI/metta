from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

import pytest
from typer.testing import CliRunner

import pufferlib.pufferl
import pufferlib.vector
from cogames import game
from cogames.main import app

runner = CliRunner()


class DummyTrainer:
    instances: List["DummyTrainer"] = []

    def __init__(self, config: dict[str, Any], vecenv: Any, policy: Any) -> None:
        self.config = config
        self.vecenv = vecenv
        self.policy = policy
        self.global_step = 0
        self.checkpoints_touched: list[Path] = []
        DummyTrainer.instances.append(self)

    def evaluate(self) -> None:  # pragma: no cover - no-op for stub
        pass

    def train(self) -> None:
        checkpoints_dir = Path(self.config["data_dir"])
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoints_dir / "dummy.ckpt"
        checkpoint_file.touch()
        self.checkpoints_touched.append(checkpoint_file)
        self.global_step = self.config["total_timesteps"]

    def print_dashboard(self) -> None:  # pragma: no cover - no-op for stub
        pass

    def close(self) -> None:  # pragma: no cover - no-op for stub
        pass


class DummyVecEnv:
    def __init__(self, env, num_envs: int) -> None:
        self.driver_env = env
        self.num_envs = num_envs
        self.num_agents = env.num_agents
        self.agents_per_batch = env.num_agents * num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space

    def close(self) -> None:  # pragma: no cover - no-op for stub
        pass


def fake_vector_make(
    env_creator,
    num_envs,
    num_workers,
    batch_size,
    backend,
    env_kwargs=None,
):
    env = env_creator()
    return DummyVecEnv(env, num_envs=num_envs)


@pytest.fixture(autouse=True)
def _reset_dummy_trainer() -> Iterator[None]:
    DummyTrainer.instances.clear()
    yield
    DummyTrainer.instances.clear()


def fake_curriculum() -> Iterable[Any]:
    from cogames.cogs_vs_clips.scenarios import make_game

    return [make_game(num_cogs=1), make_game(num_cogs=2)]


def _invoke_cli(args: Sequence[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(pufferlib.pufferl, "PuffeRL", DummyTrainer)
    monkeypatch.setattr(pufferlib.vector, "make", fake_vector_make)

    result = runner.invoke(app, ["train", *args])
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    assert DummyTrainer.instances, "trainer was not constructed"
    checkpoints_dir = Path(args[args.index("--checkpoints") + 1]) if "--checkpoints" in args else None
    if checkpoints_dir is not None:
        assert checkpoints_dir.exists()
    return result


def test_train_simple_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints_dir = tmp_path / "simple"
    run_dir = tmp_path / "simple_run"
    _invoke_cli(
        [
            "assembler_1_simple",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--num-envs",
            "1",
            "--num-workers",
            "1",
            "--batch-size",
            "4",
            "--minibatch-size",
            "4",
            "--run-dir",
            str(run_dir),
            "--checkpoints",
            str(checkpoints_dir),
        ],
        monkeypatch,
        tmp_path,
    )

    trainer = DummyTrainer.instances[0]
    assert trainer.config["use_rnn"] is False
    assert Path(trainer.config["data_dir"]).exists()


def test_train_stateful_policy_with_curriculum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints_dir = tmp_path / "stateful"
    run_dir = tmp_path / "stateful_run"

    _invoke_cli(
        [
            "machina_1",
            "--policy",
            "cogames.examples.stateful_policy.StatefulPolicy",
            "--use-rnn",
            "--curriculum",
            f"{__name__}.fake_curriculum",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--num-envs",
            "1",
            "--num-workers",
            "1",
            "--batch-size",
            "4",
            "--minibatch-size",
            "4",
            "--run-dir",
            str(run_dir),
            "--checkpoints",
            str(checkpoints_dir),
        ],
        monkeypatch,
        tmp_path,
    )

    trainer = DummyTrainer.instances[0]
    assert trainer.config["use_rnn"] is True
    assert Path(trainer.config["data_dir"]).exists()


def test_curricula_dump(tmp_path: Path) -> None:
    destination = tmp_path / "maps"
    result = runner.invoke(
        app,
        [
            "curricula",
            "--output-dir",
            str(destination),
            "--game",
            "assembler_1_simple",
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    assert (destination / "assembler_1_simple.yaml").exists()


def test_curricula_dump_with_curriculum(tmp_path: Path) -> None:
    destination = tmp_path / "maps"
    result = runner.invoke(
        app,
        [
            "curricula",
            "--output-dir",
            str(destination),
            "--game",
            "assembler_1_simple",
            "--curriculum",
            f"{__name__}.fake_curriculum",
            "--max-items",
            "2",
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    # At least one curriculum file should exist alongside the explicit game export.
    assert any(path.name.startswith("curriculum_") for path in destination.iterdir())


def test_train_uses_run_dir_curricula(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "runs" / "default"
    curricula_dir = run_dir / "curricula"
    curricula_dir.mkdir(parents=True, exist_ok=True)

    config = game.get_game("assembler_1_simple")
    game.save_game_config(config, curricula_dir / "assembler_1_simple.yaml")

    checkpoints_dir = run_dir / "checkpoints"

    _invoke_cli(
        [
            "assembler_1_simple",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--num-envs",
            "1",
            "--num-workers",
            "1",
            "--batch-size",
            "4",
            "--minibatch-size",
            "4",
            "--run-dir",
            str(run_dir),
        ],
        monkeypatch,
        tmp_path,
    )

    assert checkpoints_dir.exists()
