import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

from metta.rl.loss.base_loss import BaseLoss
from metta.rl.trainer_state import TrainerState


class Dynamics(BaseLoss):
    """The dynamics term in the Muesli loss."""

    # BaseLoss calls this method
    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        # Tell the policy that we're starting a new minibatch so it can do things like reset its memory
        policy_td = shared_loss_data["policy_td"]

        self.policy.policy.components["returns_pred"](policy_td)
        self.policy.policy.components["reward_pred"](policy_td)
        returns_pred: Tensor = policy_td["returns_pred"].to(dtype=torch.float32)
        reward_pred: Tensor = policy_td["reward_pred"].to(dtype=torch.float32)

        # need to reshape from (B, T, 1) to (B, T)
        returns_pred = einops.rearrange(returns_pred, "b t 1 -> b (t 1)")
        reward_pred = einops.rearrange(reward_pred, "b t 1 -> b (t 1)")

        # targets
        returns = shared_loss_data["sampled_mb"]["returns"]
        rewards = shared_loss_data["sampled_mb"]["rewards"]

        # The model predicts future returns and rewards.
        # We align the predictions with the future targets by slicing the tensors.
        look_ahead = self.loss_cfg.returns_step_look_ahead

        # Predict returns `look_ahead` steps into the future.
        future_returns = returns[look_ahead:]
        returns_pred_aligned = returns_pred[:-look_ahead]
        returns_loss = F.mse_loss(returns_pred_aligned, future_returns) * self.loss_cfg.returns_pred_coef

        # Predict reward at the next timestep.
        future_rewards = rewards[1:]
        reward_pred_aligned = reward_pred[:-1]
        reward_loss = F.mse_loss(reward_pred_aligned, future_rewards) * self.loss_cfg.reward_pred_coef

        self.loss_tracker["dynamics_returns_loss"].append(float(returns_loss.item()))
        self.loss_tracker["dynamics_reward_loss"].append(float(reward_loss.item()))

        return returns_loss + reward_loss, shared_loss_data
