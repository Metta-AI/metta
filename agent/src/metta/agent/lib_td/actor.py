import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from metta.common.config.config import Config


class ActorQueryConfig(Config):
    hidden_size: int = 128
    embed_dim: int = 16
    in_key: str = "hidden"
    out_key: str = "query"


class ActorQuery(nn.Module):
    """
    Projects the hidden state to a query vector.
    """

    def __init__(self, config: Optional[ActorQueryConfig] = None):
        super().__init__()
        self.config = config or ActorQueryConfig()
        self.hidden_size = self.config.hidden_size  # input_1 dim
        self.embed_dim = self.config.embed_dim  # input_2 dim (_action_embeds_)
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key

        # nn.Bilinear but hand written as nn.Parameters. As of 4-23-25, this is 10x faster than using nn.Bilinear.
        self.W = nn.Parameter(torch.Tensor(self.hidden_size, self.embed_dim).to(dtype=torch.float32))
        self._tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization"""
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)

    def forward(self, td: TensorDict):
        hidden = td[self.in_key]  # Shape: [B*TT, hidden]

        # Project hidden state to query
        query = torch.einsum("b h, h e -> b e", hidden, self.W)  # Shape: [B*TT, embed_dim]
        query = self._tanh(query)

        td[self.out_key] = query
        return td


class ActorKeyConfig(Config):
    hidden_size: int = 128
    embed_dim: int = 16
    query_key: str = "query"
    embedding_key: str = "action_embeddings"
    out_key: str = "logits"


class ActorKey(nn.Module):
    """
    Computes action scores based on a query and action embeddings (keys).
    """

    def __init__(self, config: Optional[ActorKeyConfig] = None):
        super().__init__()
        self.config = config or ActorKeyConfig()
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

    def initialize_to_environment(self, action_index_tensor: torch.Tensor, cum_action_max_params: torch.Tensor):
        self.action_index_tensor = action_index_tensor
        self.cum_action_max_params = cum_action_max_params

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None):
        query = td[self.query_key]  # Shape: [B*TT, embed_dim]
        action_embeds = td[self.embedding_key]  # Shape: [B*TT, num_actions, embed_dim]

        # Compute scores
        scores = torch.einsum("b e, b a e -> b a", query, action_embeds)  # Shape: [B*TT, num_actions]

        # Add bias
        biased_scores = scores + self.bias  # Shape: [B*TT, num_actions]

        td[self.out_key] = biased_scores
        if action is None:
            self.forward_inference(td, biased_scores)
        else:
            self.forward_training(td, action, biased_scores)
        return td

    def forward_inference(self, td: TensorDict, logits_list: torch.Tensor) -> TensorDict:
        """Forward pass for inference mode with action sampling."""
        log_probs = F.log_softmax(logits_list, dim=-1)
        action_probs = torch.exp(log_probs)

        actions = torch.multinomial(action_probs, num_samples=1).view(-1)
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        selected_log_probs = log_probs[batch_indices, actions]

        action = self._convert_logit_index_to_action(actions)

        td["actions"] = action.to(dtype=torch.int32)
        td["act_log_prob"] = selected_log_probs
        td["full_log_probs"] = log_probs

        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor, logits_list: torch.Tensor) -> TensorDict:
        """Forward pass for training mode with proper TD reshaping."""
        # CRITICAL: ComponentPolicy expects the action to be flattened already during training
        # The TD should be reshaped to match the flattened batch dimension
        if action.dim() == 3:  # (B, T, A) -> (BT, A)
            batch_size_orig, time_steps, A = action.shape
            action = action.view(batch_size_orig * time_steps, A)
            # Also flatten the TD to match
            if td.batch_dims > 1:
                td = td.reshape(td.batch_size.numel())

        action_log_probs = F.log_softmax(logits_list, dim=-1)
        action_probs = torch.exp(action_log_probs)

        action_logit_index = self._convert_action_to_logit_index(action)
        batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
        selected_log_probs = action_log_probs[batch_indices, action_logit_index]

        entropy = -(action_probs * action_log_probs).sum(dim=-1)

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
        return cumulative_sum + action_params

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices back to (action_type, action_param) pairs."""
        return self.action_index_tensor[logit_indices]
