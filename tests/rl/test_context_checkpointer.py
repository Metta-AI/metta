from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from metta.rl.training.context_checkpointer import ContextCheckpointer, ContextCheckpointerConfig


def test_context_checkpointer_forces_save_on_policy_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_manager = SimpleNamespace(checkpoint_dir=checkpoint_dir, save_trainer_state=MagicMock())
    distributed = MagicMock()
    distributed.should_checkpoint.return_value = True

    component = ContextCheckpointer(
        config=ContextCheckpointerConfig(epoch_interval=97),
        checkpoint_manager=checkpoint_manager,
        distributed_helper=distributed,
    )

    context = SimpleNamespace(latest_saved_policy_epoch=4)
    component.register(context)

    context.latest_saved_policy_epoch = 5

    with patch.object(component, "_save_state") as save_state:
        component.on_epoch_end(epoch=5)

    save_state.assert_called_once_with(force=True)


def test_context_checkpointer_save_state_clears_state(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_manager = SimpleNamespace(checkpoint_dir=checkpoint_dir, save_trainer_state=MagicMock())
    distributed = MagicMock()
    distributed.should_checkpoint.return_value = True

    component = ContextCheckpointer(
        config=ContextCheckpointerConfig(epoch_interval=10),
        checkpoint_manager=checkpoint_manager,
        distributed_helper=distributed,
    )

    class DummyStopwatch:
        def save_state(self):
            return {"elapsed_time": 1.23}

    class DummyOptimizer:
        def state_dict(self):
            return {"state": {"param": torch.ones(1)}, "param_groups": []}

    loss_mock = MagicMock()
    loss_mock.state_dict.return_value = {"buffer": torch.ones(1)}

    context = SimpleNamespace(
        epoch=7,
        agent_step=42,
        state=SimpleNamespace(stopwatch_state=None, curriculum_state=None, loss_states=None, optimizer_state=None),
        stopwatch=DummyStopwatch(),
        optimizer=DummyOptimizer(),
        curriculum=None,
        losses={"loss": loss_mock},
        latest_saved_policy_epoch=7,
    )

    component.register(context)

    component._save_state(force=True)

    checkpoint_manager.save_trainer_state.assert_called_once()
    assert context.state.optimizer_state is None
    assert context.state.loss_states == {}
    assert component._last_synced_policy_epoch == context.latest_saved_policy_epoch
