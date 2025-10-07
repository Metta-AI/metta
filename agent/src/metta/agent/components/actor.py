import math
from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions


class ActorQueryConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "actor_query"
    hidden_size: int = 512
    embed_dim: int = 16

    def make_component(self, env=None):
        return ActorQuery(config=self)


class ActorQuery(nn.Module):
    def __init__(self, config: ActorQueryConfig):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size  # input_1 dim
        self.embed_dim = self.config.embed_dim  # input_2 dim (_action_embeds_)
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key

        self.W = nn.Parameter(torch.empty(self.hidden_size, self.embed_dim, dtype=torch.float32))
        self._tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization"""
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)

    def forward(self, td: TensorDict):
        hidden = td[self.in_key]  # Shape: [..., hidden]

        # Support both flattened and multi-dimensional batches by contracting over the hidden dim.
        query = torch.einsum("... h, h e -> ... e", hidden, self.W)
        query = self._tanh(query)

        td[self.out_key] = query
        return td


class ActorKeyConfig(ComponentConfig):
    query_key: str
    embedding_key: str
    out_key: str
    name: str = "actor_key"
    hidden_size: int = 128
    embed_dim: int = 16

    def make_component(self, env=None):
        return ActorKey(config=self)


class ActorKey(nn.Module):
    """
    Computes action scores based on a query and action embeddings (keys).
    """

    def __init__(self, config: ActorKeyConfig):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.embed_dim = self.config.embed_dim
        self.query_key = self.config.query_key
        self.embedding_key = self.config.embedding_key
        self.out_key = self.config.out_key

        self.bias = nn.Parameter(torch.Tensor(1).to(dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization for bias"""
        if self.bias is not None:
            # The input to this layer is the query dim
            bound = 1 / math.sqrt(self.embed_dim)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, td: TensorDict):
        query = td[self.query_key]  # Shape: [..., embed_dim]
        action_embeds = td[self.embedding_key]  # Shape: [..., num_actions, embed_dim]

        # Compute scores across any batch shape
        scores = torch.einsum("... e, ... a e -> ... a", query, action_embeds)

        # Add bias
        biased_scores = scores + self.bias  # Shape: [B*TT, num_actions]

        td[self.out_key] = biased_scores
        return td


class ActionProbsConfig(ComponentConfig):
    in_key: str
    name: str = "action_probs"

    def make_component(self, env=None):
        return ActionProbs(config=self)


class ActionProbs(nn.Module):
    """
    Computes action scores based on a query and action embeddings (keys).
    """

    def __init__(self, config: ActionProbsConfig):
        super().__init__()
        self.config = config

    def initialize_to_environment(
        self,
        env: Any,
        device,
    ) -> None:
        action_max_params = list(env.max_action_args)
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        self.action_index_tensor = torch.tensor(
            [[idx, j] for idx, max_param in enumerate(action_max_params) for j in range(max_param + 1)],
            device=device,
            dtype=torch.int32,
        )

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        if action is None:
            return self.forward_inference(td)
        else:
            return self.forward_training(td, action)

    def forward_inference(self, td: TensorDict) -> TensorDict:
        logits = td[self.config.in_key]
        """Forward pass for inference mode with action sampling."""
        action_logit_index, selected_log_probs, _, full_log_probs = sample_actions(logits)

        action = self._convert_logit_index_to_action(action_logit_index)

        td["actions"] = action.to(dtype=torch.int32)
        td["act_log_prob"] = selected_log_probs
        td["full_log_probs"] = full_log_probs

        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """Forward pass for training mode with proper TD reshaping."""
        # CRITICAL: ComponentPolicy expects the action to be flattened already during training
        # The TD should be reshaped to match the flattened batch dimension
        logits = td[self.config.in_key]
        if action.dim() == 3:  # (B, T, A) -> (BT, A)
            batch_size_orig, time_steps, A = action.shape
            action = action.view(batch_size_orig * time_steps, A)
            # Also flatten the TD to match
            if td.batch_dims > 1:
                td = td.reshape(td.batch_size.numel())

        action_logit_index = self._convert_action_to_logit_index(action)
        selected_log_probs, entropy, action_log_probs = evaluate_actions(logits, action_logit_index)

        # Store in flattened TD (will be reshaped by caller if needed)
        td["act_log_prob"] = selected_log_probs
        td["entropy"] = entropy
        td["full_log_probs"] = action_log_probs

        # ComponentPolicy reshapes the TD after training forward based on td["batch"] and td["bptt"]
        # The reshaping happens in ComponentPolicy.forward() after forward_training()
        if "batch" in td.keys() and "bptt" in td.keys():
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            td = td.reshape(batch_size, bptt_size)

        return td

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_type_numbers + action_params

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices back to (action_type, action_param) pairs."""
        return self.action_index_tensor[logit_indices]
