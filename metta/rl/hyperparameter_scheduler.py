import logging


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
        new_lr = trainer_cfg.optimizer.learning_rate * (cfg.learning_rate_decay**progress)
        optimizer.param_groups[0]["lr"] = new_lr
        updates["learning_rate"] = new_lr

    if cfg.ppo_clip_decay < 1.0:
        updates["ppo_clip_coef"] = trainer_cfg.ppo.clip_coef * (cfg.ppo_clip_decay**progress)

    if current_step % 10000 == 0 and updates:
        params = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
        logger.info(f"Step {current_step}: {params}")

    return updates
