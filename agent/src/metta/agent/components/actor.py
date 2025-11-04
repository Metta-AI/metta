import math
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from metta.rl.training import GameRules


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

    def initialize_to_environment(
        self,
        env: GameRules,
        device: torch.device,
    ) -> None:
        from gymnasium.spaces import Discrete

        action_space = env.action_space
        if not isinstance(action_space, Discrete):
            msg = f"ActionProbs expects a Discrete action space, got {type(action_space).__name__}"
            raise TypeError(msg)

        self.num_actions = int(action_space.n)

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        if action is None:
            return self.forward_inference(td)
        else:
            return self.forward_training(td, action)

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """Forward pass for inference mode with action sampling."""
        logits = td[self.config.in_key]
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

    def make_component(self, env: GameRules | None = None):
        if env is None:
            raise ValueError("ActorHeadConfig requires GameRules to determine action dimensions")
        return ActorHead(config=self, env=env)


class ActorHead(nn.Module):
    """Simple linear head that maps hidden features to environment logits."""

    def __init__(self, config: ActorHeadConfig, env: GameRules):
        super().__init__()
        self.config = config
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        num_actions = int(env.action_space.n)

        linear = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.input_dim, num_actions),
            std=self.config.layer_init_std,
        )
        self._module = TDM(linear, in_keys=[self.in_key], out_keys=[self.out_key])

    def forward(self, td: TensorDict) -> TensorDict:
        return self._module(td)
