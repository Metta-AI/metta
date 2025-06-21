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

        # Auxiliary losses for representation learning
        self.sensory_decoder_loss_sum = 0.0
        self.latent_decoder_loss_sum = 0.0
        self.node_perturbation_loss_sum = 0.0
        self.contrastive_loss_sum = 0.0

        self.explained_variance = 0.0
        self.minibatches_processed = 0

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
            # Auxiliary losses
            "sensory_decoder_loss": self.sensory_decoder_loss_sum / n,
            "latent_decoder_loss": self.latent_decoder_loss_sum / n,
            "node_perturbation_loss": self.node_perturbation_loss_sum / n,
            "contrastive_loss": self.contrastive_loss_sum / n,
            "explained_variance": self.explained_variance,
        }
