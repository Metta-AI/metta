from types import SimpleNamespace

from metta.rl.trainer_config import TrainerConfig, UpdateEpochAutoTunerConfig
from metta.rl.training.update_epochs_tuner import UpdateEpochAutoTuner


class FakeStopwatch:
    def __init__(self) -> None:
        self._elapsed = 0.0

    def advance(self, delta: float) -> None:
        self._elapsed += delta

    def get_elapsed(self, name: str | None = None) -> float:  # noqa: ARG002 - API compatibility
        return self._elapsed


class MasterOnlyDistributed:
    def __init__(self) -> None:
        self._last = None

    def is_master(self) -> bool:
        return True

    def broadcast_from_master(self, obj):
        self._last = obj
        return obj


def test_update_epochs_autotuner_explores_and_finds_best_value() -> None:
    autotune_cfg = UpdateEpochAutoTunerConfig(
        enabled=True,
        min_update_epochs=1,
        max_update_epochs=3,
        step_size=1,
        evaluation_epochs=1,
        warmup_epochs=0,
        cooldown_epochs=0,
        min_relative_improvement=0.0,
        metrics_window=2,
    )
    trainer_cfg = TrainerConfig(update_epochs=1, update_epochs_autotune=autotune_cfg)

    context = SimpleNamespace(
        config=trainer_cfg,
        agent_step=0,
        epoch=0,
        stopwatch=FakeStopwatch(),
        distributed=MasterOnlyDistributed(),
    )

    tuner = UpdateEpochAutoTuner(autotune_cfg)
    tuner.register(context)

    # Initial epoch establishes baseline timing without making adjustments.
    context.agent_step = 1_000
    context.epoch = 1
    context.stopwatch.advance(1.0)
    tuner.on_epoch_end(context.epoch)
    assert context.config.update_epochs == 1

    # First measured epoch at update_epochs=1 triggers exploration upwards.
    context.agent_step = 2_000
    context.epoch = 2
    context.stopwatch.advance(1.0)
    tuner.on_epoch_end(context.epoch)
    assert context.config.update_epochs == 2

    # High throughput at update_epochs=2 encourages another exploration step.
    context.agent_step = 3_600
    context.epoch = 3
    context.stopwatch.advance(1.0)
    tuner.on_epoch_end(context.epoch)
    assert context.config.update_epochs == 3

    # Poor throughput at update_epochs=3 causes the tuner to revert to the best-known value (2).
    context.agent_step = 4_500
    context.epoch = 4
    context.stopwatch.advance(1.0)
    tuner.on_epoch_end(context.epoch)
    assert context.config.update_epochs == 2
