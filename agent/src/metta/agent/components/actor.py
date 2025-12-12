import math
from typing import Optional

import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


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

        query = torch.einsum("b h, h e -> b e", hidden, self.W)  # Shape: [B*TT, embed_dim]
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
        query = td[self.query_key]  # Shape: [B*TT, embed_dim]
        action_embeds = td[self.embedding_key]  # Shape: [B*TT, num_actions, embed_dim]

        # Compute scores
        scores = torch.einsum("b e, b a e -> b a", query, action_embeds)  # Shape: [B*TT, num_actions]

        # Add bias
        biased_scores = scores + self.bias  # Shape: [B*TT, num_actions]

        td[self.out_key] = biased_scores
        return td


class ActionProbsConfig(ComponentConfig):
    in_key: str
    name: str = "action_probs"
    # Mask everything past the first 21 actions (5 base + 16 TRAINING_VIBES) everywhere.
    max_action_index: int | None = 21

    def make_component(self, env=None):
        return ActionProbs(config=self)


class ActionProbs(nn.Module):
    """
    Computes action scores based on a query and action embeddings (keys).
    """

    def __init__(self, config: ActionProbsConfig):
        super().__init__()
        self.config = config
        self.num_actions = 0
        # Persist the trained action cap in checkpoints so masking stays consistent.
        self.register_buffer(
            "_max_action_index_buf",
            torch.tensor(21, dtype=torch.int32),
            persistent=True,
        )

    def _ensure_initialized(self) -> None:
        if self.num_actions <= 0:
            raise RuntimeError("ActionProbs not initialized; call initialize_to_environment before forward.")

    def initialize_to_environment(
        self,
        env: PolicyEnvInterface,
        device: torch.device,
    ) -> None:
        action_space = env.action_space
        if not isinstance(action_space, Discrete):
            msg = f"ActionProbs expects a Discrete action space, got {type(action_space).__name__}"
            raise TypeError(msg)

        self.num_actions = int(action_space.n)

        # Always use the canonical training cap of 21 actions; store so it round-trips.
        self._max_action_index_buf.fill_(21)

    def _mask_logits_if_needed(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask logits past the first 21 actions (or the stored cap)."""
        max_action_index = int(self._max_action_index_buf.item())

        max_idx = int(max_action_index)
        if max_idx <= 0:
            raise ValueError(f"max_action_index must be positive, got {max_idx}")

        max_idx = min(max_idx, logits.size(-1))
        mask_value = torch.finfo(logits.dtype).min

        # Keep logits finite so log_softmax never returns NaN (which would break torch.multinomial).
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=mask_value)

        if max_idx < logits.size(-1):
            logits = logits.clone()
            logits[..., max_idx:] = mask_value

        # If every allowed action was invalid and got clamped to mask_value, fall back to a
        # zeroed valid slice so we still produce a proper distribution over allowed actions only.
        valid_slice = logits[..., :max_idx]
        all_masked = (valid_slice == mask_value).all(dim=-1)
        if all_masked.any():
            logits = logits.clone()
            logits[all_masked, :max_idx] = 0.0

        return logits

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Allow checkpoints that lack the mask buffer and default to 21."""
        buf_key = prefix + "_max_action_index_buf"
        if buf_key in missing_keys:
            missing_keys.remove(buf_key)
        if buf_key not in state_dict:
            # Default to the training cap (21 actions: 5 base + 16 TRAINING_VIBES)
            state_dict[buf_key] = torch.tensor(21, dtype=torch.int32)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        if action is None:
            return self.forward_inference(td)
        else:
            return self.forward_training(td, action)

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """Forward pass for inference mode with action sampling."""
        logits = td[self.config.in_key]

        self._ensure_initialized()
        logits = self._mask_logits_if_needed(logits)
        action_logit_index, selected_log_probs, _, full_log_probs = sample_actions(logits)

        td["actions"] = action_logit_index.to(dtype=torch.int32)
        td["act_log_prob"] = selected_log_probs
        td["full_log_probs"] = full_log_probs

        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """Forward pass for training mode with proper TD reshaping."""
        # CRITICAL: ComponentPolicy expects the action to be flattened already during training
        # The TD should be reshaped to match the flattened batch dimension
        logits = td[self.config.in_key]
        if action.dim() == 3:
            batch_size_orig, time_steps, _ = action.shape
            action = action.view(batch_size_orig * time_steps, -1)
            # Also flatten the TD to match
            if td.batch_dims > 1:
                td = td.reshape(td.batch_size.numel())

        if action.dim() == 2 and action.size(1) == 1:
            action = action.view(-1)

        if action.dim() != 1:
            raise ValueError(f"Expected flattened action indices, got shape {tuple(action.shape)}")

        action_logit_index = action.to(dtype=torch.long)
        self._ensure_initialized()
        logits = self._mask_logits_if_needed(logits)
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


class ActorHeadConfig(ComponentConfig):
    in_key: str
    out_key: str
    input_dim: int
    layer_init_std: float = 1.0
    name: str = "actor_head"

    def make_component(self, env: PolicyEnvInterface | None = None):
        if env is None:
            raise ValueError("ActorHeadConfig requires PolicyEnvInterface to determine action dimensions")
        return ActorHead(config=self, env=env)


class ActorHead(nn.Module):
    """Simple linear head that maps hidden features to environment logits."""

    def __init__(self, config: ActorHeadConfig, env: PolicyEnvInterface):
        super().__init__()
        self.config = config
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.num_actions = int(env.action_space.n)

        linear = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.input_dim, self.num_actions),
            std=self.config.layer_init_std,
        )
        self._module = TDM(linear, in_keys=[self.in_key], out_keys=[self.out_key])

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Pad actor_head weights/bias if checkpoint has fewer actions than env."""
        weight_key = prefix + "_module.module.weight"
        bias_key = prefix + "_module.module.bias"

        weight = state_dict.get(weight_key)
        bias = state_dict.get(bias_key)

        if weight is not None and bias is not None and weight.shape[0] < self.num_actions:
            pad = self.num_actions - weight.shape[0]
            # Pad weights with zeros; pad biases with -inf to keep zero prob.
            weight_pad = torch.zeros(pad, weight.shape[1], dtype=weight.dtype, device=weight.device)
            bias_pad = torch.full((pad,), -1e9, dtype=bias.dtype, device=bias.device)
            state_dict[weight_key] = torch.cat([weight, weight_pad], dim=0)
            state_dict[bias_key] = torch.cat([bias, bias_pad], dim=0)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, td: TensorDict) -> TensorDict:
        return self._module(td)
