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
        self.importance = 0.0
        self.minibatches_processed = 0

    def to_dict(self) -> dict[str, float]:
        """Convert losses to dictionary with proper averages"""
        n = max(1, self.minibatches_processed)

        return {
            "policy_loss": self.policy_loss / n,
            "value_loss": self.value_loss / n,
            "entropy": self.entropy / n,
            "old_approx_kl": self.old_approx_kl / n,
            "approx_kl": self.approx_kl / n,
            "clipfrac": self.clipfrac / n,
            "l2_reg_loss": self.l2_reg_loss / n,
            "l2_init_loss": self.l2_init_loss / n,
            "ks_action_loss": self.ks_action_loss / n,
            "ks_value_loss": self.ks_value_loss / n,
            "importance": self.importance / n,
            "explained_variance": self.explained_variance,
        }
