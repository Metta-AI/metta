from types import SimpleNamespace

import torch

from metta.rl.trainer import Trainer


class _DummyLoss:
    def __init__(self):
        self.loss_profiles = None


def test_action_supervisor_profiles_are_resolved(monkeypatch):
    # Stub minimal trainer pieces
    trainer_cfg = SimpleNamespace()
    trainer_cfg.losses = SimpleNamespace(supervisor=SimpleNamespace(profiles=["teach"]), _configs=lambda: {})
    trainer_cfg.loss_profiles = {"teach": None}

    trainer = object.__new__(Trainer)
    trainer._cfg = trainer_cfg

    losses = {"action_supervisor": _DummyLoss()}
    loss_profile_lookup = {"teach": 7}

    trainer._assign_loss_profiles(losses, loss_profile_lookup)

    assert losses["action_supervisor"].loss_profiles == {7}
