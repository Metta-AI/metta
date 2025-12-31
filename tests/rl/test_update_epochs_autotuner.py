from types import SimpleNamespace

from metta.rl.trainer_config import TrainerConfig, UpdateEpochAutoTunerConfig
from metta.rl.training.update_epochs_tuner import UpdateEpochAutoTuner


class MasterOnlyDistributed:
    def __init__(self) -> None:
        self._last = None

    def is_master(self) -> bool:
        return True

    def broadcast_from_master(self, obj):
        self._last = obj
        return obj


def test_update_epochs_autotuner_tracks_kl_signal() -> None:
    autotune_cfg = UpdateEpochAutoTunerConfig(
        min_update_epochs=1,
        max_update_epochs=3,
        step_size=1,
        evaluation_epochs=1,
        warmup_epochs=0,
        cooldown_epochs=0,
        target_kl=0.02,
        kl_tolerance=0.25,
        max_clipfrac=0.2,
    )
    trainer_cfg = TrainerConfig(update_epochs=1, update_epochs_autotune=autotune_cfg)

    context = SimpleNamespace(
        config=trainer_cfg,
        agent_step=0,
        epoch=0,
        latest_graph_stats={},
        distributed=MasterOnlyDistributed(),
    )

    tuner = UpdateEpochAutoTuner(autotune_cfg)
    tuner.register(context)

    # Low KL encourages reusing the batch once.
    context.latest_graph_stats = {"ppo_actor/approx_kl": 0.002, "ppo_actor/clipfrac": 0.05}
    tuner.on_epoch_end(epoch=1)
    assert context.config.update_epochs == 2

    # KL near target keeps the current value steady.
    context.latest_graph_stats = {"ppo_actor/approx_kl": 0.018, "ppo_actor/clipfrac": 0.07}
    tuner.on_epoch_end(epoch=2)
    assert context.config.update_epochs == 2

    # Excessive clip fraction nudges the tuner back down.
    context.latest_graph_stats = {"ppo_actor/approx_kl": 0.03, "ppo_actor/clipfrac": 0.4}
    tuner.on_epoch_end(epoch=3)
    assert context.config.update_epochs == 1
