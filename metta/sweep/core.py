"""Sweep parameter helpers for Ray-based sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ray import tune


@dataclass(frozen=True)
class ParameterSpec:
    """A single sweep parameter: dotted path + Ray Tune domain."""

    path: str
    space: Any


class SweepParameters:
    """Common parameter presets and canonical sets for sweeps."""

    # Individual optimizer parameters
    LEARNING_RATE = ParameterSpec(
        path="trainer.optimizer.learning_rate",
        space=tune.loguniform(1e-5, 1e-2),
    )

    ADAM_EPS = ParameterSpec(
        path="trainer.optimizer.eps",
        space=tune.loguniform(1e-8, 1e-4),
    )

    # Individual PPO loss parameters
    PPO_CLIP_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.clip_coef",
        space=tune.uniform(0.05, 0.3),
    )

    PPO_ENT_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.ent_coef",
        space=tune.loguniform(1e-4, 3e-2),
    )

    PPO_GAE_LAMBDA = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.gae_lambda",
        space=tune.uniform(0.8, 0.99),
    )

    PPO_VF_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.vf_coef",
        space=tune.uniform(0.1, 1.0),
    )

    @classmethod
    def adam_optimizer_hypers(cls) -> list[ParameterSpec]:
        """Canonical set of Adam optimizer hyperparameters."""
        return [
            cls.LEARNING_RATE,
            ParameterSpec("trainer.optimizer.beta1", tune.uniform(0.85, 0.99)),
            ParameterSpec("trainer.optimizer.beta2", tune.uniform(0.95, 0.9999)),
            cls.ADAM_EPS,
            ParameterSpec("trainer.optimizer.weight_decay", tune.choice([0.0, 1e-6, 1e-5, 1e-4])),
        ]

    @classmethod
    def sgd_optimizer_hypers(cls) -> list[ParameterSpec]:
        """Canonical set of SGD optimizer hyperparameters."""
        return [
            cls.LEARNING_RATE,
            ParameterSpec("trainer.optimizer.momentum", tune.uniform(0.8, 0.99)),
            ParameterSpec("trainer.optimizer.weight_decay", tune.choice([0.0, 1e-6, 1e-5, 1e-4])),
        ]

    @classmethod
    def muon_optimizer_hypers(cls) -> list[ParameterSpec]:
        """Canonical set of Muon optimizer hyperparameters."""
        return [
            ParameterSpec("trainer.optimizer.learning_rate", tune.loguniform(5e-4, 5e-2)),  # Higher LR range for Muon
            ParameterSpec("trainer.optimizer.beta1", tune.uniform(0.9, 0.99)),  # High momentum for Muon
            ParameterSpec("trainer.optimizer.beta2", tune.uniform(0.95, 0.9999)),  # Second moment estimate
            ParameterSpec("trainer.optimizer.eps", tune.loguniform(1e-8, 1e-5)),
            ParameterSpec("trainer.optimizer.weight_decay", tune.choice([0, 1, 2, 5, 10])),  # Int values for Muon
        ]

    @classmethod
    def ppo_loss_hypers(cls, include_advanced: bool = True) -> list[ParameterSpec]:
        """Canonical set of PPO loss hyperparameters.

        Args:
            include_advanced: If True, includes V-trace and PER parameters
        """
        params = [
            ParameterSpec("trainer.losses.loss_configs.ppo.clip_coef", tune.uniform(0.005, 0.3)),
            ParameterSpec("trainer.losses.loss_configs.ppo.ent_coef", tune.loguniform(1e-4, 1e-1)),
            ParameterSpec("trainer.losses.loss_configs.ppo.gae_lambda", tune.uniform(0.8, 0.99)),
            ParameterSpec("trainer.losses.loss_configs.ppo.gamma", tune.uniform(0.95, 0.999)),
            ParameterSpec("trainer.losses.loss_configs.ppo.max_grad_norm", tune.uniform(0.1, 1.0)),
            ParameterSpec("trainer.losses.loss_configs.ppo.vf_clip_coef", tune.uniform(0.0, 0.5)),
            ParameterSpec("trainer.losses.loss_configs.ppo.vf_coef", tune.uniform(0.1, 1.0)),
            ParameterSpec("trainer.losses.loss_configs.ppo.l2_reg_loss_coef", tune.choice([0.0, 1e-6, 1e-5, 1e-4])),
            ParameterSpec("trainer.losses.loss_configs.ppo.l2_init_loss_coef", tune.choice([0.0, 1e-6, 1e-5, 1e-4])),
            ParameterSpec("trainer.losses.loss_configs.ppo.norm_adv", tune.choice([True, False])),
            ParameterSpec("trainer.losses.loss_configs.ppo.clip_vloss", tune.choice([True, False])),
            ParameterSpec("trainer.losses.loss_configs.ppo.target_kl", tune.choice([None, 0.01, 0.05, 0.1])),
        ]

        if include_advanced:
            # V-trace parameters
            params.extend(
                [
                    ParameterSpec("trainer.losses.loss_configs.ppo.vtrace.rho_clip", tune.uniform(0.5, 2.0)),
                    ParameterSpec("trainer.losses.loss_configs.ppo.vtrace.c_clip", tune.uniform(0.5, 2.0)),
                ]
            )
            # Prioritized experience replay parameters
            params.extend(
                [
                    ParameterSpec(
                        "trainer.losses.loss_configs.ppo.prioritized_experience_replay.prio_alpha",
                        tune.uniform(0.0, 1.0),
                    ),
                    ParameterSpec(
                        "trainer.losses.loss_configs.ppo.prioritized_experience_replay.prio_beta0",
                        tune.uniform(0.4, 1.0),
                    ),
                ]
            )

        return params

    @classmethod
    def training_hypers(cls) -> list[ParameterSpec]:
        """Canonical set of training hyperparameters (batch sizes, epochs, etc)."""
        return [
            ParameterSpec("trainer.batch_size", tune.choice([131_072, 262_144, 524_288])),
            ParameterSpec("trainer.minibatch_size", tune.choice([8_192, 16_384, 32_768])),
            ParameterSpec("trainer.bptt_horizon", tune.choice([16, 32, 64, 128])),
            ParameterSpec("trainer.update_epochs", tune.randint(1, 6)),
        ]

    @classmethod
    def full_sweep(cls, optimizer: str = "adam") -> list[ParameterSpec]:
        """Complete canonical sweep including optimizer, PPO, and training params.

        Args:
            optimizer: Which optimizer to sweep ("adam", "sgd", or "muon")
        """
        if optimizer == "adam":
            opt_params = cls.adam_optimizer_hypers()
        elif optimizer == "sgd":
            opt_params = cls.sgd_optimizer_hypers()
        elif optimizer == "muon":
            opt_params = cls.muon_optimizer_hypers()
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Choose 'adam', 'sgd', or 'muon'")

        return [
            *opt_params,
            *cls.ppo_loss_hypers(include_advanced=True),
            *cls.training_hypers(),
        ]


# Convenience alias for cleaner imports
SP = SweepParameters

__all__ = [
    "ParameterSpec",
    "SweepParameters",
    "SP",
]
