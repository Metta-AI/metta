import logging
from typing import Dict, Optional, Tuple

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.agalite import AGaLiTe

logger = logging.getLogger(__name__)


class Agalite(nn.Module):
    """AGaLiTe (Approximate Gated Linear Transformer) policy for Metta environments."""

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

        # Create the AGaLiTe core
        self.agalite = AGaLiTe(
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

        self.to(self.device)

    def forward(self, td: TensorDict, state: Optional[Dict] = None, action=None) -> TensorDict:
        """Forward pass through AGaLiTe policy."""

        # Handle BPTT reshaping
        if td.batch_dims > 1:
            B, TT = td.batch_size
            total_batch = B * TT
            td = td.reshape(total_batch)
            # After reshaping, create tensors matching the new flattened batch size
            td.set("bptt", torch.full((total_batch,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((total_batch,), B, device=td.device, dtype=torch.long))
        else:
            batch_size = td.batch_size[0] if hasattr(td.batch_size, "__getitem__") else td.batch_size
            td.set("bptt", torch.full((batch_size,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((batch_size,), batch_size, device=td.device, dtype=torch.long))

        observations = td["env_obs"].to(self.device)

        # Initialize or deserialize state
        if state is None:
            agalite_memory = self._initialize_agalite_memory()
        elif isinstance(state, torch.Tensor):
            # State is passed as a serialized tensor (from td["state"] or previous forward pass)
            agalite_memory = self._deserialize_agalite_memory(state)
        elif isinstance(state, dict):
            # State is passed as a dictionary
            if "agalite_memory_serialized" in state:
                agalite_memory = self._deserialize_agalite_memory(state["agalite_memory_serialized"])
            else:
                agalite_memory = state.get("agalite_memory", self._initialize_agalite_memory())
        else:
            # Fallback to initialization
            agalite_memory = self._initialize_agalite_memory()

        # Encode observations
        hidden = self.encode_observations(observations)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare terminations (use zeros if not provided)
        terminations = td.get("terminations", torch.zeros(B * TT, device=self.device))

        hidden = hidden.view(B * TT, -1)  # Flatten for AGaLiTe
        agalite_out, new_agalite_memory = self.agalite(hidden, terminations, agalite_memory)

        # Decode actions and value
        logits_list, value = self.decode_actions(agalite_out)

        # Sample actions and compute probabilities
        actions, log_probs, entropies, full_log_probs = self._sample_actions(logits_list)

        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)
        actions_tensor = actions_tensor.to(dtype=torch.int32)

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
            if td.batch_dims > 1:
                td = td.reshape(B, TT)

        # Update state - serialize AGaLiTe memory as a single tensor (similar to LSTM state storage)
        # NOTE: Current AGaLiTe implementation processes all batch elements through a single model
        # with shared memory, which is not ideal for independent batch processing.
        # For now, we skip storing state to avoid dimension mismatches.
        # TODO: Switch to BatchedAGaLiTe for proper per-batch-element memory handling
        pass

        return td

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations into features for AGaLiTe."""
        observations = observations.to(self.device)
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        # Process token observations
        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        x_coords = ((coords_byte >> 4) & 0x0F).long()
        y_coords = (coords_byte & 0x0F).long()
        atr_indices = observations[..., 1].long()
        atr_values = observations[..., 2].float()

        # Create box observations
        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=self.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = torch.arange(B * TT, device=self.device).unsqueeze(-1).expand_as(atr_values)
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

    def _initialize_state(self, batch_size: int) -> Dict:
        """Initialize the state dictionary."""
        return {"agalite_memory": self._initialize_agalite_memory()}

    def _initialize_agalite_memory(self) -> Dict[str, Tuple]:
        """Initialize AGaLiTe memory."""
        return AGaLiTe.initialize_memory(self.n_layers, self.n_heads, self.d_head, self.eta, self.r, self.device)

    def _serialize_agalite_memory(self, memory_dict: Dict[str, Tuple]) -> torch.Tensor:
        """
        Serialize AGaLiTe memory dictionary into a single tensor.

        Memory structure per layer: (tilde_k_prev, tilde_v_prev, s_prev, tick)
        We concatenate all tensors from all layers into a single flat tensor.
        """
        all_tensors = []
        for layer_idx in range(1, self.n_layers + 1):
            layer_memory = memory_dict[f"layer_{layer_idx}"]
            # Flatten each tensor in the tuple and concatenate
            for tensor in layer_memory:
                all_tensors.append(tensor.flatten())

        # Concatenate all flattened tensors
        return torch.cat(all_tensors, dim=0)

    def _deserialize_agalite_memory(self, serialized: torch.Tensor) -> Dict[str, Tuple]:
        """
        Deserialize a single tensor back into AGaLiTe memory dictionary.

        Reconstructs the original dictionary structure with proper tensor shapes.
        """
        memory_dict = {}
        offset = 0

        # Calculate sizes for each tensor component
        tilde_k_size = self.r * self.n_heads * self.eta * self.d_head
        tilde_v_size = self.r * self.n_heads * self.d_head
        s_size = self.n_heads * self.eta * self.d_head
        tick_size = 1

        for layer_idx in range(1, self.n_layers + 1):
            # Extract and reshape each component
            tilde_k_prev = serialized[offset : offset + tilde_k_size].reshape(
                self.r, self.n_heads, self.eta * self.d_head
            )
            offset += tilde_k_size

            tilde_v_prev = serialized[offset : offset + tilde_v_size].reshape(self.r, self.n_heads, self.d_head)
            offset += tilde_v_size

            s_prev = serialized[offset : offset + s_size].reshape(self.n_heads, self.eta * self.d_head)
            offset += s_size

            tick = serialized[offset : offset + tick_size]
            offset += tick_size

            memory_dict[f"layer_{layer_idx}"] = (tilde_k_prev, tilde_v_prev, s_prev, tick)

        return memory_dict

    def clip_weights(self):
        """Clip weights to prevent large updates."""
        for p in self.parameters():
            p.data.clamp_(-1, 1)
