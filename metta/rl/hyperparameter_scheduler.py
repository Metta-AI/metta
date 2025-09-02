import logging
import math


def _decay_value(initial: float, decay_rate: float, progress: float, schedule_type: str = "exponential") -> float:
    if schedule_type == "cosine":
        return initial * (1 + math.cos(math.pi * progress)) / 2
    elif schedule_type == "linear":
        return initial * (1 - progress)
    else:
        return initial * (decay_rate**progress)


def step_hyperparameters(
    trainer_cfg, optimizer, current_step: int, total_timesteps: int, logger=None
) -> dict[str, float]:
    logger = logger or logging.getLogger(__name__)
    cfg = trainer_cfg.hyperparameter_scheduler

    if not getattr(cfg, "enabled", False):
        return {}

    progress = min(current_step / max(total_timesteps, 1), 1.0)
    updates = {}

    if cfg.learning_rate_decay < 1.0:
        new_lr = _decay_value(trainer_cfg.optimizer.learning_rate, cfg.learning_rate_decay, progress, cfg.schedule_type)
        optimizer.param_groups[0]["lr"] = new_lr
        updates["learning_rate"] = new_lr

    if cfg.ppo_clip_decay < 1.0:
        updates["ppo_clip_coef"] = _decay_value(
            trainer_cfg.ppo.clip_coef, cfg.ppo_clip_decay, progress, cfg.schedule_type
        )

    if cfg.ppo_ent_coef_decay < 1.0:
        updates["ppo_ent_coef"] = _decay_value(
            trainer_cfg.ppo.ent_coef, cfg.ppo_ent_coef_decay, progress, cfg.schedule_type
        )

    if current_step % 10000 == 0 and updates:
        params = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
        logger.info(f"Step {current_step}: {params}")

    return updates
