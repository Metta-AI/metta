"""Shared CMPO loss configuration helpers."""

from metta.rl.loss.cmpo import CMPOConfig
from metta.rl.loss.losses import LossesConfig


def cmpo_losses() -> LossesConfig:
    losses = LossesConfig()
    losses.ppo_actor.enabled = False
    losses.ppo_critic.enabled = False
    losses.cmpo = CMPOConfig(enabled=True)
    return losses


__all__ = ["cmpo_losses"]
