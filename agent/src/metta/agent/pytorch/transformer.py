"""Transformer agent for Metta."""

import logging
import math
import warnings
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.transformer_module import TransformerModule
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    def __init__(
        self,
        env,
        input_size: int = 128,
        hidden_size: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.out_width = env.obs_width if hasattr(env, "obs_width") else 11
        self.out_height = env.obs_height if hasattr(env, "obs_height") else 11
        self.num_layers = max(env.feature_normalizations.keys()) + 1 if hasattr(env, "feature_normalizations") else 25

        self.cnn1 = init_layer(nn.Conv2d(self.num_layers, 64, 5, 3), std=1.0)
        self.cnn2 = init_layer(nn.Conv2d(64, 64, 3, 1), std=1.0)

        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(torch.zeros(1, self.num_layers, self.out_width, self.out_height)))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 128), std=1.0)
        self.encoded_obs = init_layer(nn.Linear(128, input_size), std=1.0)

        self._transformer = TransformerModule(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )

        self.critic_1 = init_layer(nn.Linear(hidden_size, 1024), std=1.0)
        self.value_head = init_layer(nn.Linear(1024, 1), std=0.1)
        self.actor_1 = init_layer(nn.Linear(hidden_size, 512), std=0.5)
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()
        self.action_embed_dim = 16
        self.actor_hidden_dim = 512
        self._init_bilinear_actor()

        max_values = [1.0] * self.num_layers
        if hasattr(env, "feature_normalizations"):
            for fid, norm in env.feature_normalizations.items():
                if fid < self.num_layers:
                    max_values[fid] = norm if norm > 0 else 1.0
        self.register_buffer("max_vec", torch.tensor(max_values, dtype=torch.float32)[None, :, None, None])
        self.active_action_names = []
        self.num_active_actions = 100

    def _initialize_action_embeddings(self):
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            self.action_embeddings.weight.mul_(0.1 / torch.max(torch.abs(self.action_embeddings.weight)))

    def _init_bilinear_actor(self):
        self.actor_W = nn.Parameter(torch.Tensor(1, self.actor_hidden_dim, self.action_embed_dim).float())
        self.actor_bias = nn.Parameter(torch.Tensor(1).float())
        bound = 1 / math.sqrt(self.actor_hidden_dim) if self.actor_hidden_dim > 0 else 0
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)

    def activate_action_embeddings(self, full_action_names: list[str], device):
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, x):
        x = x / self.max_vec
        x = F.relu(self.cnn2(F.relu(self.cnn1(x))))
        x = F.relu(self.encoded_obs(F.relu(self.fc1(self.flatten(x)))))
        return x

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]
        B_TT = B * TT
        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
        assert observations.shape[-1] == 3

        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()
        valid_tokens = coords_byte != 0xFF
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr
        if (valid_tokens & ~valid_atr).any():
            warnings.warn(f"Found obs attribute indices >= {self.num_layers}, ignoring", stacklevel=2)

        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + x_coord_indices * self.out_height + y_coord_indices
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        return self.network_forward(box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height))

    def decode_actions(self, hidden: torch.Tensor, batch_size: int) -> tuple:
        value = self.value_head(torch.tanh(self.critic_1(hidden)))
        actor_features = F.relu(self.actor_1(hidden))
        action_embeds = self.action_embeddings.weight[: self.num_active_actions].unsqueeze(0).expand(batch_size, -1, -1)
        num_actions = action_embeds.shape[1]
        actor_reshaped = actor_features.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, self.actor_hidden_dim)
        action_embeds_reshaped = action_embeds.reshape(-1, self.action_embed_dim)
        query = torch.tanh(torch.einsum("n h, k h e -> n k e", actor_reshaped, self.actor_W))
        logits = (torch.einsum("n k e, n e -> n k", query, action_embeds_reshaped) + self.actor_bias).reshape(
            batch_size, num_actions
        )
        return logits, value

    def transformer(self, hidden: torch.Tensor, terminations: torch.Tensor = None, memory: dict = None):
        return self._transformer(hidden), None

    def initialize_memory(self, batch_size: int) -> dict:
        return {}


class Transformer(PyTorchAgentMixin, TransformerWrapper):
    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 128,
        hidden_size: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        **kwargs,
    ):
        mixin_params = self.extract_mixin_params(kwargs)
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_causal_mask=use_causal_mask,
                use_gating=use_gating,
            )
        super().__init__(env, policy, hidden_size)
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]
        if state is None:
            state = {"transformer_memory": None, "hidden": None}
        B, TT = (
            (observations.shape[0], observations.shape[1]) if observations.dim() == 4 else (observations.shape[0], 1)
        )
        if observations.dim() == 4 and td.batch_dims > 1:
            td = td.reshape(B * TT)
        self.set_tensordict_fields(td, observations)
        hidden = self.policy.encode_observations(observations, state)
        hidden = hidden.view(B, TT, -1).transpose(0, 1) if TT > 1 else hidden.unsqueeze(0)
        hidden, _ = self.policy.transformer(hidden, None, state.get("transformer_memory"))
        hidden = hidden.transpose(0, 1).reshape(B * TT, -1) if TT > 1 else hidden.squeeze(0)
        logits, values = self.policy.decode_actions(hidden, B * TT)
        if action is None:
            td = self.forward_inference(td, logits, values)
        else:
            if values.dim() == 2:
                values = values.reshape(-1)
            td = self.forward_training(td, action, logits, values)
        return td
