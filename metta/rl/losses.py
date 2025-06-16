class Losses:
    def __init__(self):
        self.zero()

    def zero(self):
        """Reset all loss values to 0.0"""
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy = 0.0
        self.old_approx_kl = 0.0
        self.approx_kl = 0.0
        self.clipfrac = 0.0
        self.explained_variance = 0.0
        self.l2_reg_loss = 0.0
        self.l2_init_loss = 0.0
        self.ks_action_loss = 0.0
        self.ks_value_loss = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert losses to dictionary for stats/logging"""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}
