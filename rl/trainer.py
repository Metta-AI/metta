# Kickstarter modifications
self.ppo_loss_avg_duration_steps = trainer_cfg.kickstart.ppo_loss_avg_duration_steps
self.ppo_loss_history = checkpoint.extra_args.get("ppo_loss_history", [])
self.ppo_loss_avg = checkpoint.extra_args.get("ppo_loss_avg", 0.0)
if self.kickstarter.enabled:
    self.kickstarter.ks_loss_history = checkpoint.extra_args.get("ks_loss_history", [])
    self.kickstarter.initial_ks_to_ppo_ratio = checkpoint.extra_args.get("initial_ks_to_ppo_ratio", None)

# Optimizer
optimizer_type = getattr(trainer_cfg.optimizer, "type", "adam")
assert optimizer_type in ("adam", "muon"), f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}"


def _checkpoint_trainer(self):
    if not self._master:
        return

    self._checkpoint_policy()

    extra_args = {}
    if self.kickstarter.enabled and self.kickstarter.teacher_uri is not None:
        extra_args["teacher_pr_uri"] = self.kickstarter.teacher_uri

    # Kickstarter modifications
    extra_args["ppo_loss_history"] = self.ppo_loss_history
    extra_args["ppo_loss_avg"] = self.ppo_loss_avg
    if self.kickstarter.enabled:
        extra_args["ks_loss_history"] = self.kickstarter.ks_loss_history
        extra_args["initial_ks_to_ppo_ratio"] = self.kickstarter.initial_ks_to_ppo_ratio

    checkpoint = TrainerCheckpoint(
        agent_step=self.agent_step,
        epoch=self.epoch,
        # ... existing code ...
    )
