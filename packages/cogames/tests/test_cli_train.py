from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

import pytest
from typer.testing import CliRunner

from cogames import train as train_module
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


def fake_vector_make(env_creator, num_envs, num_workers, batch_size, backend, env_kwargs):
    cfg_iterator = env_kwargs["cfg_iterator"]
    env = env_creator(cfg_iterator, buf=None, seed=None)
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
    monkeypatch.setattr(train_module.pufferl, "PuffeRL", DummyTrainer)
    monkeypatch.setattr(train_module.pufferlib.vector, "make", fake_vector_make)

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
            "--checkpoints",
            str(checkpoints_dir),
        ],
        monkeypatch,
        tmp_path,
    )

    trainer = DummyTrainer.instances[0]
    assert trainer.config["use_rnn"] is True
    assert Path(trainer.config["data_dir"]).exists()
