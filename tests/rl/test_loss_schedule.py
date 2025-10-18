from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite

from metta.rl.loss.loss import Loss
from metta.rl.loss.loss_config import LossConfig, LossPhaseSchedule, LossSchedule
from mettagrid.base_config import Config


class DummyPolicy:
    def get_agent_experience_spec(self) -> Composite:
        return Composite()

    def reset_memory(self) -> None:  # pragma: no cover - required by Loss base class
        return


class DummyLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.register_state_attr("counter")

    def run_train(self, shared_loss_data: TensorDict, context, mb_idx: int):
        self.counter += 1
        return torch.tensor(float(self.counter), device=self.device), shared_loss_data, False


class DummyLossConfig(Config):
    schedule: LossSchedule | None = None

    def create(self, policy, trainer_cfg, env, device, instance_name, loss_config):
        return DummyLoss(policy, trainer_cfg, env, device, instance_name, loss_config)


class DummyContext:
    def __init__(self, epoch: int):
        self.epoch = epoch


def _make_shared_loss_data() -> TensorDict:
    return TensorDict({}, batch_size=[])


def test_loss_respects_begin_end_schedule():
    schedule = LossSchedule(train=LossPhaseSchedule(begin_at_epoch=2, end_at_epoch=4))
    cfg = DummyLossConfig(schedule=schedule)
    loss = DummyLoss(DummyPolicy(), object(), object(), torch.device("cpu"), "dummy", cfg)

    # Epoch 0 and 1 should skip training
    zero_loss, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=0), mb_idx=0)
    assert zero_loss.item() == 0.0
    assert loss.counter == 0

    zero_loss, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=1), mb_idx=0)
    assert zero_loss.item() == 0.0
    assert loss.counter == 0

    # Epochs 2 and 3 should run
    out, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=2), mb_idx=0)
    assert out.item() == 1.0
    out, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=3), mb_idx=0)
    assert out.item() == 2.0

    # Epoch 4 should be excluded (end is exclusive)
    zero_loss, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=4), mb_idx=0)
    assert zero_loss.item() == 0.0
    assert loss.counter == 2


def test_loss_cycle_schedule():
    schedule = LossSchedule(train=LossPhaseSchedule(begin_at_epoch=0, cycle_length=3, active_in_cycle=[1, 3]))
    cfg = DummyLossConfig(schedule=schedule)
    loss = DummyLoss(DummyPolicy(), object(), object(), torch.device("cpu"), "dummy", cfg)

    # Epochs 0 (cycle step 1) and 2 (cycle step 3) should run
    out, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=0), mb_idx=0)
    assert out.item() == 1.0
    zero_loss, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=1), mb_idx=0)
    assert zero_loss.item() == 0.0
    out, _, _ = loss.train(_make_shared_loss_data(), DummyContext(epoch=2), mb_idx=0)
    assert out.item() == 2.0


def test_loss_state_dict_roundtrip():
    cfg = DummyLossConfig(schedule=None)
    loss = DummyLoss(DummyPolicy(), object(), object(), torch.device("cpu"), "dummy", cfg)
    loss.train(_make_shared_loss_data(), DummyContext(epoch=0), mb_idx=0)

    state = loss.state_dict()
    assert state["counter"] == 1

    loss.counter = 0
    loss.load_state_dict(state)
    assert loss.counter == 1


def test_loss_config_applies_schedule_to_entries():
    loss_cfg = DummyLossConfig()
    schedule = LossSchedule(train=LossPhaseSchedule(begin_at_epoch=1))
    aggregate = LossConfig(loss_configs={"dummy": loss_cfg}, loss_schedules={"dummy": schedule})

    losses = aggregate.init_losses(
        DummyPolicy(), trainer_cfg=SimpleNamespace(), env=object(), device=torch.device("cpu")
    )
    dummy_loss = losses["dummy"]
    assert dummy_loss.loss_cfg.schedule == schedule


def test_invalid_schedule_raises():
    invalid_schedule = LossSchedule(train=LossPhaseSchedule(begin_at_epoch=3, end_at_epoch=2))
    cfg = DummyLossConfig(schedule=invalid_schedule)

    with pytest.raises(ValueError):
        DummyLoss(DummyPolicy(), object(), object(), torch.device("cpu"), "dummy", cfg)
