class Losses:
    def __init__(self):
        self.zero()

    def zero(self):
        """Reset all loss values and counters"""
        # Cumulative values
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.old_approx_kl_sum = 0.0
        self.approx_kl_sum = 0.0
        self.clipfrac_sum = 0.0
        self.l2_reg_loss_sum = 0.0
        self.l2_init_loss_sum = 0.0
        self.ks_action_loss_sum = 0.0
        self.ks_value_loss_sum = 0.0

        # Special case - this is computed once at the end
        self.explained_variance = 0.0

        # Counter for actual minibatches processed
        self.minibatches_processed = 0

    def to_dict(self) -> dict[str, float]:
        """Convert losses to dictionary with proper averages"""
        n = max(1, self.minibatches_processed)

        return {
            "policy_loss": self.policy_loss_sum / n,
            "value_loss": self.value_loss_sum / n,
            "entropy": self.entropy_sum / n,
            "old_approx_kl": self.old_approx_kl_sum / n,
            "approx_kl": self.approx_kl_sum / n,
            "clipfrac": self.clipfrac_sum / n,
            "l2_reg_loss": self.l2_reg_loss_sum / n,
            "l2_init_loss": self.l2_init_loss_sum / n,
            "ks_action_loss": self.ks_action_loss_sum / n,
            "ks_value_loss": self.ks_value_loss_sum / n,
            "explained_variance": self.explained_variance,
        }
