import logging
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

# this is where lstm hidden state is
from metta.agent.policy_state import PolicyState


from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.util.losses import compute_ppo_losses

logger = logging.getLogger(__name__)

def get_future_indices(
    minibatch: Dict[str, Tensor],
    trainer_cfg: Any,
    device: torch.device,
) -> Tensor:
    # u = torch.rand(batch_size, device=self.device)
    # gamma_tensor = torch.tensor(self.gamma, device=self.device)
    # delta = torch.floor(torch.log(1 - u) / torch.log(gamma_tensor)).long()

    # # Clamp to BPTT horizon
    # delta = torch.clamp(delta, min=1, max=bptt_horizon - 1)

    # # Compute future indices
    # future_indices = current_indices + delta
    # return future_indices

    u = torch.rand(minibatch["obs"].shape[0], device=device)
    gamma_tensor = torch.tensor(trainer_cfg.gamma, device=device)
    delta = torch.floor(torch.log(1 - u) / torch.log(gamma_tensor)).long()
    delta = torch.clamp(delta, min=1, max=trainer_cfg.bptt_horizon - 1)
    future_indices = minibatch["indices"] + delta

    return future_indices

def sample_negative_indices(
    minibatch: Dict[str, Tensor],
    trainer_cfg: Any,
    device: torch.device,
) -> Tensor:



    return torch.ones(1)

def compute_infonce_loss(
    minibatch: Dict[str, Tensor],
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
    device: torch.device,
    policy: torch.nn.Module,
    experience: Experience,
):
    batch_size = minibatch["obs"].shape[0]
    pass

def positives_negatives():
    pass
