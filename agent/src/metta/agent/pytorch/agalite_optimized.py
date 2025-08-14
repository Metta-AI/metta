import logging
from typing import Dict, Optional, Tuple

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.agalite_batched import BatchedAGaLiTe

logger = logging.getLogger(__name__)


class AgaliteOptimized(nn.Module):
    """Optimized AGaLiTe policy for GPU training with proper batching and state management."""

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        n_layers: int = 4,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        eta: int = 4,
        r: int = 8,
        reset_on_terminate: bool = True,
        dropout: float = 0.1,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = env.single_action_space
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.eta = eta
        self.r = r
        self.hidden_size = hidden_size

        # Initialize observation parameters
        self.out_width, self.out_height, self.num_layers = 11, 11, 22

        # Create the observation encoder (CNN + self encoder)
        self.cnn_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, 128, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(128, d_model // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, d_model // 2)),
            nn.ReLU(),
        )

        # Use the optimized batched AGaLiTe
        self.agalite = BatchedAGaLiTe(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )

        # Create action heads - one for each discrete action
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(d_model, n), std=0.01) for n in self.action_space.nvec]
        )
        self.value = pufferlib.pytorch.layer_init(nn.Linear(d_model, 1), std=1)

        # Register buffer for observation normalization
        max_vec = torch.tensor(
            [9, 1, 1, 10, 3, 254, 1, 1, 235, 8, 9, 250, 29, 1, 1, 8, 1, 1, 6, 3, 1, 2],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        # State memory buffer
        self._memory_cache = {}

        self.to(self.device)

    def forward(self, td: TensorDict, state: Optional[Dict] = None, action=None) -> TensorDict:
        """Forward pass through AGaLiTe policy with optimized batching."""

        # Extract batch dimensions
        original_shape = td.batch_size
        if td.batch_dims > 1:
            B, TT = td.batch_size
            total_batch = B * TT
            td = td.reshape(total_batch)
        else:
            B = td.batch_size[0] if hasattr(td.batch_size, "__getitem__") else td.batch_size
            TT = 1
            total_batch = B

        observations = td["env_obs"].to(self.device)

        # Encode observations
        hidden = self.encode_observations(observations)

        # Get or initialize memory
        if state is not None and "agalite_memory" in state:
            # Deserialize state from tensor
            memory = self._deserialize_memory(state["agalite_memory"], B)
        else:
            # Initialize new memory
            memory = BatchedAGaLiTe.initialize_memory(
                batch_size=B,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                d_head=self.d_head,
                eta=self.eta,
                r=self.r,
                device=self.device,
            )

        # Get terminations
        terminations = td.get("terminations", torch.zeros(total_batch, device=self.device))

        # Reshape for AGaLiTe: (B*TT, d_model) -> (TT, B, d_model)
        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
            terminations = terminations.view(B, TT).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)  # (1, B, d_model)
            terminations = terminations.unsqueeze(0)  # (1, B)

        # Forward through AGaLiTe
        agalite_out, new_memory = self.agalite(hidden, terminations, memory)

        # CRITICAL: Detach memory for experience replay to prevent gradient accumulation
        new_memory = self._detach_memory(new_memory)

        # Reshape output back: (TT, B, d_model) -> (B*TT, d_model)
        if TT > 1:
            agalite_out = agalite_out.transpose(0, 1).reshape(total_batch, -1)
        else:
            agalite_out = agalite_out.squeeze(0)

        # Decode actions and value
        logits_list, value = self.decode_actions(agalite_out)

        # Sample actions and compute probabilities
        actions, log_probs, entropies, full_log_probs = self._sample_actions(logits_list)

        # Prepare action tensor
        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)
        actions_tensor = actions_tensor.to(dtype=torch.int32)

        # Store outputs in TensorDict
        if action is None:
            # Inference mode
            td["actions"] = actions_tensor
            td["act_log_prob"] = log_probs.mean(dim=-1)
            td["values"] = value.flatten()
            td["full_log_probs"] = full_log_probs
        else:
            # Training mode
            td["act_log_prob"] = log_probs.mean(dim=-1)
            td["entropy"] = entropies.sum(dim=-1)
            td["value"] = value.flatten()
            td["full_log_probs"] = full_log_probs

        # Serialize and store memory state
        serialized_memory = self._serialize_memory(new_memory)
        td["agalite_memory"] = serialized_memory

        # Reshape back if needed
        if original_shape != td.batch_size:
            td = td.reshape(original_shape)

        return td

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations into features for AGaLiTe."""
        B = observations.shape[0]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            B = observations.shape[0]

        # Process token observations
        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        x_coords = ((coords_byte >> 4) & 0x0F).long()
        y_coords = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        # Create box observations
        box_obs = torch.zeros(
            (B, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=self.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = torch.arange(B, device=self.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[batch_idx[valid_tokens], atr_indices[valid_tokens], x_coords[valid_tokens], y_coords[valid_tokens]] = (
            atr_values[valid_tokens]
        )

        # Normalize and encode
        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.cnn_encoder(features)

        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Decode hidden states into action logits and value."""
        return [head(hidden) for head in self.actor], self.value(hidden)

    def _sample_actions(self, logits_list: list[torch.Tensor]):
        """Sample discrete actions from logits."""
        actions, selected_log_probs, entropies, full_log_probs = [], [], [], []
        max_actions = max(logits.shape[1] for logits in logits_list)

        for logits in logits_list:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            action = torch.multinomial(probs, 1).squeeze(-1)
            batch_idx = torch.arange(action.shape[0], device=action.device)

            selected_log_prob = log_probs[batch_idx, action]
            entropy = -(probs * log_probs).sum(dim=-1)

            actions.append(action)
            selected_log_probs.append(selected_log_prob)
            entropies.append(entropy)
            pad_width = max_actions - log_probs.shape[1]
            full_log_probs.append(F.pad(log_probs, (0, pad_width), value=float("-inf")))

        return (
            actions,
            torch.stack(selected_log_probs, dim=-1),
            torch.stack(entropies, dim=-1),
            torch.stack(full_log_probs, dim=-1),
        )

    def _serialize_memory(self, memory_dict: Dict[str, Tuple]) -> torch.Tensor:
        """Serialize memory dictionary into a single tensor."""
        tensors = []
        for layer_idx in range(1, self.n_layers + 1):
            layer_memory = memory_dict[f"layer_{layer_idx}"]
            for tensor in layer_memory:
                # Flatten all dimensions except batch
                batch_size = tensor.shape[0]
                flattened = tensor.view(batch_size, -1)
                tensors.append(flattened)

        return torch.cat(tensors, dim=1)

    def _deserialize_memory(self, serialized: torch.Tensor, batch_size: int) -> Dict[str, Tuple]:
        """Deserialize a tensor back into memory dictionary."""
        memory_dict = {}
        offset = 0

        # Calculate sizes for each tensor component
        tilde_k_size = self.r * self.n_heads * self.eta * self.d_head
        tilde_v_size = self.r * self.n_heads * self.d_head
        s_size = self.n_heads * self.eta * self.d_head
        tick_size = 1

        for layer_idx in range(1, self.n_layers + 1):
            tilde_k_prev = serialized[:, offset : offset + tilde_k_size].reshape(
                batch_size, self.r, self.n_heads, self.eta * self.d_head
            )
            offset += tilde_k_size

            tilde_v_prev = serialized[:, offset : offset + tilde_v_size].reshape(
                batch_size, self.r, self.n_heads, self.d_head
            )
            offset += tilde_v_size

            s_prev = serialized[:, offset : offset + s_size].reshape(batch_size, self.n_heads, self.eta * self.d_head)
            offset += s_size

            tick = serialized[:, offset : offset + tick_size]
            offset += tick_size

            memory_dict[f"layer_{layer_idx}"] = (tilde_k_prev, tilde_v_prev, s_prev, tick)

        return memory_dict

    def _detach_memory(self, memory_dict: Dict[str, Tuple]) -> Dict[str, Tuple]:
        """Detach memory tensors to prevent gradient accumulation."""
        detached_memory = {}
        for key, layer_memory in memory_dict.items():
            detached_memory[key] = tuple(tensor.detach() for tensor in layer_memory)
        return detached_memory

    def reset_memory(self) -> None:
        """Reset cached memory."""
        self._memory_cache.clear()

    def get_memory(self) -> dict:
        """Get current memory state."""
        return self._memory_cache.copy()

    # PufferLib compatibility
    def forward_eval(self, observations: torch.Tensor, state=None):
        """Forward function for inference (PufferLib compatibility)."""
        batch_size = observations.shape[0]
        td = TensorDict(
            {
                "env_obs": observations,
                "terminations": torch.zeros(batch_size, device=observations.device),
            },
            batch_size=batch_size,
        )

        if state is not None:
            td["agalite_memory"] = state

        output_td = self.forward(td)

        logits = output_td.get("full_log_probs", torch.zeros(batch_size, 2))
        values = output_td.get("values", output_td.get("value", torch.zeros(batch_size)))

        return logits, values

    def forward_train(self, observations: torch.Tensor, state=None):
        """Forward function for training (PufferLib compatibility)."""
        return self.forward_eval(observations, state)
