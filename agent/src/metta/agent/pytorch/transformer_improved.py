"""Transformer-XL agent implementation for reinforcement learning."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.transformer_module import TransformerModule
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.base import (
    bilinear_actor_forward,
    init_bilinear_actor,
    initialize_action_embeddings,
)
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin


class TransformerXLPolicy(nn.Module):
    def __init__(
        self,
        env,
        input_size: int = 256,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        memory_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.out_width = getattr(env, "obs_width", 11)
        self.out_height = getattr(env, "obs_height", 11)
        self.num_layers = max(env.feature_normalizations.keys()) + 1 if hasattr(env, "feature_normalizations") else 25

        self.cnn1 = init_layer(nn.Conv2d(self.num_layers, 64, 5, 3), std=1.0)
        self.cnn2 = init_layer(nn.Conv2d(64, 128, 3, 1), std=1.0)

        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(torch.zeros(1, self.num_layers, self.out_width, self.out_height)))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()
        self.fc1 = init_layer(nn.Linear(self.flattened_size, 512), std=1.0)
        # Produce feature vector of size `input_size` (may differ from hidden_size)
        self.encoded_obs = init_layer(nn.Linear(512, input_size), std=1.0)

        # Ensure the transformer always receives vectors of size `hidden_size`.
        # If input_size != hidden_size, add a projection; otherwise use identity.
        if input_size != hidden_size:
            self.to_hidden = init_layer(nn.Linear(input_size, hidden_size), std=1.0)
        else:
            self.to_hidden = nn.Identity()

        self.critic_hidden_dim = 1024
        self.actor_hidden_dim = 512
        self.action_embed_dim = 16

        self.critic_1 = init_layer(nn.Linear(hidden_size, self.critic_hidden_dim), std=math.sqrt(2))
        self.value_head = init_layer(nn.Linear(self.critic_hidden_dim, 1), std=1.0)

        self.actor_1 = init_layer(nn.Linear(hidden_size, self.actor_hidden_dim), std=1.0)
        self.action_embeddings = nn.Embedding(100, self.action_embed_dim)
        initialize_action_embeddings(self.action_embeddings)
        self.actor_W, self.actor_bias = init_bilinear_actor(self.actor_hidden_dim, self.action_embed_dim)

        self._transformer = TransformerModule(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            memory_len=memory_len,
            dropout=dropout,
            dropatt=dropout,
            pre_lnorm=True,
        )

        max_values = [1.0] * self.num_layers
        if hasattr(env, "feature_normalizations"):
            for fid, norm in env.feature_normalizations.items():
                if fid < self.num_layers:
                    max_values[fid] = norm if norm > 0 else 1.0
        self.register_buffer("max_vec", torch.tensor(max_values, dtype=torch.float32)[None, :, None, None])

        self.active_action_names = []
        self.num_active_actions = 100

    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment, setting up action embeddings to match the available actions."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)

    def network_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations into `hidden_size` vectors for the transformer."""
        x = x / self.max_vec
        x = F.relu(self.cnn2(F.relu(self.cnn1(x))))
        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.encoded_obs(x))
        x = self.to_hidden(x)
        return x

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        if observations.dim() == 4:
            B, T = observations.shape[:2]
            observations = observations.reshape(B * T, *observations.shape[2:])

        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        valid_tokens = coords_byte != 0xFF
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + x_coord_indices * self.out_height + y_coord_indices
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        box_flat = torch.zeros(
            (observations.shape[0], self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)

        return self.network_forward(box_flat.view(-1, self.num_layers, self.out_width, self.out_height))

    def decode_actions(self, hidden: torch.Tensor, batch_size: int = None) -> tuple:
        """Standard GTrXL action/value decoding."""
        critic_features = torch.tanh(self.critic_1(hidden))
        values = self.value_head(critic_features).squeeze(-1)

        actor_features = F.relu(self.actor_1(hidden))
        action_embeds = self.action_embeddings.weight[: self.num_active_actions]
        action_embeds = action_embeds.unsqueeze(0).expand(hidden.shape[0], -1, -1)

        logits = bilinear_actor_forward(
            actor_features,
            action_embeds,
            self.actor_W,
            self.actor_bias,
            self.actor_hidden_dim,
            self.action_embed_dim,
        )

        return logits, values

    def transformer(self, hidden: torch.Tensor, terminations: torch.Tensor = None, memory: dict = None):
        output, new_memory = self._transformer(hidden, memory)
        if terminations is not None and new_memory is not None:
            hidden_states = new_memory.get("hidden_states") if isinstance(new_memory, dict) else None
            if hidden_states:
                done_mask = terminations[-1].to(torch.bool)
                if done_mask.any():
                    for layer_mem in hidden_states:
                        if layer_mem is None or layer_mem.numel() == 0:
                            continue
                        layer_mem[:, done_mask, :] = 0
        return output, new_memory

    def initialize_memory(self, batch_size: int) -> dict:
        return self._transformer.initialize_memory(batch_size)


class TransformerImproved(PyTorchAgentMixin, TransformerWrapper):
    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 256,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        memory_len: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        mixin_params = self.extract_mixin_params(kwargs)
        if policy is None:
            policy = TransformerXLPolicy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                memory_len=memory_len,
                dropout=dropout,
            )
        super().__init__(env, policy, hidden_size)
        self.init_mixin(**mixin_params)

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]
        if state is None:
            state = {"transformer_memory": None, "hidden": None}

        B = observations.shape[0]
        TT = observations.shape[1] if observations.dim() == 4 else 1

        if observations.dim() == 4 and td.batch_dims > 1:
            td = td.reshape(B * TT)

        self.set_tensordict_fields(td, observations)

        memory_before = state.get("transformer_memory")
        hidden = self.policy.encode_observations(observations, state)

        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is None and truncateds is not None:
            dones = truncateds
            truncateds = None

        if dones is not None:
            dones = dones.to(torch.bool)
            if truncateds is not None:
                dones = dones | truncateds.to(torch.bool)
            if TT > 1:
                terminations = dones.reshape(B, TT).transpose(0, 1)
            else:
                terminations = dones.reshape(B).unsqueeze(0)
        else:
            base = torch.zeros(B, dtype=torch.bool)
            if TT > 1:
                terminations = base.unsqueeze(0).expand(TT, -1)
            else:
                terminations = base.unsqueeze(0)

        terminations = terminations.to(hidden.device, dtype=hidden.dtype).contiguous()
        state["terminations"] = terminations[-1:].detach()

        hidden, memory = self.policy.transformer(hidden, terminations, state.get("transformer_memory"))
        normalized_memory = self._normalize_memory(memory)
        if normalized_memory is not None:
            state["transformer_memory"] = self._detach_memory(normalized_memory)
        else:
            state["transformer_memory"] = None

        segment_indices = td.get("_segment_indices", None)
        segment_pos = td.get("_segment_pos", None)
        if segment_indices is not None and segment_pos is not None:
            self._record_segment_memory(segment_indices, segment_pos, memory_before)

        if TT > 1:
            hidden = hidden.transpose(0, 1).reshape(B * TT, -1)
        else:
            hidden = hidden.squeeze(0)

        logits, values = self._decode_actions(hidden, B * TT)
        if action is None:
            td = self.forward_inference(td, logits, values)
        else:
            if values.dim() == 2:
                values = values.reshape(-1)
            td = self.forward_training(td, action, logits, values)

        return td
