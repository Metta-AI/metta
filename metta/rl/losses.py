class Losses:
    def __init__(self):
        self.zero()

    def zero(self):
        """Reset all loss values to 0.0"""
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.approx_kl_sum = 0.0
        self.clipfrac_sum = 0.0
        self.l2_reg_loss_sum = 0.0
        self.l2_init_loss_sum = 0.0
        self.ks_action_loss_sum = 0.0
        self.ks_value_loss_sum = 0.0
        self.importance_sum = 0.0
        self.current_logprobs_sum = 0.0
        self.explained_variance = 0.0
        self.minibatches_processed = 0
        self.contrastive_loss_sum = 0.0
        self.contrastive_infonce_sum = 0.0
        self.contrastive_logsumexp_sum = 0.0
        self.contrastive_pos_sim_sum = 0.0
        self.contrastive_neg_sim_sum = 0.0

    def stats(self) -> dict[str, float]:
        """Convert losses to dictionary with proper averages"""
        n = max(1, self.minibatches_processed)

        return {
            "policy_loss": self.policy_loss_sum / n,
            "value_loss": self.value_loss_sum / n,
            "entropy": self.entropy_sum / n,
            "approx_kl": self.approx_kl_sum / n,
            "clipfrac": self.clipfrac_sum / n,
            "l2_reg_loss": self.l2_reg_loss_sum / n,
            "l2_init_loss": self.l2_init_loss_sum / n,
            "ks_action_loss": self.ks_action_loss_sum / n,
            "ks_value_loss": self.ks_value_loss_sum / n,
            "importance": self.importance_sum / n,
            "explained_variance": self.explained_variance,
            "current_logprobs": self.current_logprobs_sum / n,
            "contrastive_loss": self.contrastive_loss_sum / n,
            "contrastive_infonce": self.contrastive_infonce_sum / n,
            "contrastive_logsumexp": self.contrastive_logsumexp_sum / n,
            "contrastive_pos_sim": self.contrastive_pos_sim_sum / n,
            "contrastive_neg_sim": self.contrastive_neg_sim_sum / n,
        }
