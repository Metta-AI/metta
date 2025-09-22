import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

logger = logging.getLogger(__name__)

_MAX_LOGGED_INDICES = 5
_MAX_LOGGED_VALUES = 10
_MAX_LOGGED_TD_KEYS = 25


def _snapshot_td_keys(td: TensorDict) -> list[str]:
    """Return a truncated, stringified snapshot of TD keys for logging."""
    keys: list[str] = []
    try:
        raw_keys = list(td.keys())
    except TypeError:
        raw_keys = [key for key in td.keys()]

    total = len(raw_keys)
    limit = min(total, _MAX_LOGGED_TD_KEYS)
    for idx in range(limit):
        keys.append(repr(raw_keys[idx]))
    if total > limit:
        keys.append(f"...( +{total - limit} more )")
    return keys


def _collect_row_context(td: TensorDict, batch_index: int) -> dict[str, Any]:
    """Capture lightweight context tensors for a problematic batch index."""
    context: dict[str, Any] = {}
    collected = 0
    for key in td.keys():
        if collected >= 5:
            break
        try:
            value = td.get(key)
        except KeyError:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if value.dim() == 0 or value.shape[0] <= batch_index:
            continue
        with torch.no_grad():
            sample_tensor = value[batch_index]
            flattened = sample_tensor.reshape(-1)
            sample_values = flattened[:_MAX_LOGGED_VALUES].detach().cpu().tolist()
        context[repr(key)] = {
            "shape": list(sample_tensor.shape),
            "dtype": str(sample_tensor.dtype),
            "sample_values": sample_values,
        }
        collected += 1
    return context


def _log_and_sanitize(
    *,
    tensor: torch.Tensor,
    name: str,
    component: str,
    td: TensorDict,
) -> tuple[torch.Tensor, bool]:
    """
    Inspect `tensor`, emit diagnostics for NaN/Inf, and return a sanitized tensor.

    Returns a tuple of (possibly sanitized tensor, whether an anomaly was detected).
    """

    if tensor is None:
        return tensor, False

    invalid_mask = ~torch.isfinite(tensor)
    if not invalid_mask.any():
        return tensor, False

    invalid_count = int(invalid_mask.sum().item())
    nan_count = int(torch.isnan(tensor).sum().item())
    posinf_count = int(torch.isposinf(tensor).sum().item())
    neginf_count = int(torch.isneginf(tensor).sum().item())

    detached = tensor.detach()
    sanitized_view = torch.nan_to_num(detached, nan=0.0, posinf=0.0, neginf=0.0)
    finite_min = float(sanitized_view.min().item()) if sanitized_view.numel() else 0.0
    finite_max = float(sanitized_view.max().item()) if sanitized_view.numel() else 0.0
    finite_abs_max = float(sanitized_view.abs().max().item()) if sanitized_view.numel() else 0.0

    invalid_indices = invalid_mask.nonzero(as_tuple=False)[:_MAX_LOGGED_INDICES].detach().cpu().tolist()
    offending_batch_index = None
    if invalid_indices:
        first_index = invalid_indices[0]
        if len(first_index) > 0:
            offending_batch_index = first_index[0]
    batch_context: dict[str, Any] | None = None
    if offending_batch_index is not None:
        batch_context = _collect_row_context(td, int(offending_batch_index))

    logger.error(
        "[%s] Invalid values detected in %s: shape=%s dtype=%s total_invalid=%d nan=%d posinf=%d neginf=%d "
        "finite_min=%s finite_max=%s finite_abs_max=%s sample_indices=%s td_keys=%s batch_context=%s",
        component,
        name,
        list(tensor.shape),
        tensor.dtype,
        invalid_count,
        nan_count,
        posinf_count,
        neginf_count,
        finite_min,
        finite_max,
        finite_abs_max,
        invalid_indices,
        _snapshot_td_keys(td),
        batch_context,
    )

    sanitized = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized, True


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
        hidden = td[self.in_key]  # Shape: [B*TT, hidden]
        hidden, hidden_invalid = _log_and_sanitize(
            tensor=hidden,
            name=f"{self.__class__.__name__}.input[{self.in_key}]",
            component="ActorQuery",
            td=td,
        )
        if hidden_invalid:
            td[self.in_key] = hidden

        query = torch.einsum("b h, h e -> b e", hidden, self.W)  # Shape: [B*TT, embed_dim]
        query = self._tanh(query)

        query, query_invalid = _log_and_sanitize(
            tensor=query,
            name=f"{self.__class__.__name__}.output[{self.out_key}]",
            component="ActorQuery",
            td=td,
        )
        if query_invalid:
            td[self.out_key] = query

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
        query = td[self.query_key]  # Shape: [B*TT, embed_dim]
        action_embeds = td[self.embedding_key]  # Shape: [B*TT, num_actions, embed_dim]

        query, query_invalid = _log_and_sanitize(
            tensor=query,
            name=f"{self.__class__.__name__}.query[{self.query_key}]",
            component="ActorKey",
            td=td,
        )
        if query_invalid:
            td[self.query_key] = query

        action_embeds, embeds_invalid = _log_and_sanitize(
            tensor=action_embeds,
            name=f"{self.__class__.__name__}.embeds[{self.embedding_key}]",
            component="ActorKey",
            td=td,
        )
        if embeds_invalid:
            td[self.embedding_key] = action_embeds

        bias_tensor = self.bias
        _, bias_invalid = _log_and_sanitize(
            tensor=bias_tensor,
            name=f"{self.__class__.__name__}.bias",
            component="ActorKey",
            td=td,
        )
        if bias_invalid:
            with torch.no_grad():
                self.bias.copy_(torch.nan_to_num(self.bias, nan=0.0, posinf=0.0, neginf=0.0))

        # Compute scores
        scores = torch.einsum("b e, b a e -> b a", query, action_embeds)  # Shape: [B*TT, num_actions]

        scores, _ = _log_and_sanitize(
            tensor=scores,
            name=f"{self.__class__.__name__}.scores",
            component="ActorKey",
            td=td,
        )

        # Add bias
        biased_scores = scores + self.bias  # Shape: [B*TT, num_actions]

        biased_scores, _ = _log_and_sanitize(
            tensor=biased_scores,
            name=f"{self.__class__.__name__}.biased_scores[{self.out_key}]",
            component="ActorKey",
            td=td,
        )

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
        raw_logits = logits

        row_all_neg_inf = torch.isneginf(raw_logits).all(dim=-1)
        if row_all_neg_inf.any():
            offending_rows = torch.nonzero(row_all_neg_inf, as_tuple=False)[:5].view(-1).cpu().tolist()
            logger.error(
                "All-logits-masked rows encountered before sampling: rows=%s",
                offending_rows,
            )

        logits, logits_invalid = _log_and_sanitize(
            tensor=logits,
            name=f"{self.__class__.__name__}.logits[{self.config.in_key}]",
            component="ActionProbs",
            td=td,
        )
        if logits_invalid:
            td[self.config.in_key] = logits

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
        raw_logits = logits
        if action.dim() == 3:  # (B, T, A) -> (BT, A)
            batch_size_orig, time_steps, A = action.shape
            action = action.view(batch_size_orig * time_steps, A)
            # Also flatten the TD to match
            if td.batch_dims > 1:
                td = td.reshape(td.batch_size.numel())

        row_all_neg_inf = torch.isneginf(raw_logits).all(dim=-1)
        if row_all_neg_inf.any():
            offending_rows = torch.nonzero(row_all_neg_inf, as_tuple=False)[:5].view(-1).cpu().tolist()
            logger.error(
                "All-logits-masked rows encountered before training evaluation: rows=%s",
                offending_rows,
            )

        logits, logits_invalid = _log_and_sanitize(
            tensor=logits,
            name=f"{self.__class__.__name__}.logits[{self.config.in_key}]",
            component="ActionProbs",
            td=td,
        )
        if logits_invalid:
            td[self.config.in_key] = logits

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
