"""
Transformer wrapper for recurrent policies in PufferLib/Metta infrastructure.
Similar to LSTMWrapper but handles transformer-style memory states and full sequences.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SegmentMemoryRecord:
    """Snapshot of transformer memory captured at the start of a replay segment."""

    segment_index: int
    memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]


class TransformerWrapper(nn.Module):
    """
    Wrapper for transformer-based recurrent policies.

    Key differences from LSTM:
    - Processes entire BPTT sequences as context (not step-by-step)
    - Maintains complex memory state (not just h/c)
    - Handles termination-aware memory resets
    """

    def __setstate__(self, state):
        """Ensure transformer memory states are properly initialized after loading from checkpoint.

        Similar to LSTM's __setstate__, prevents batch size mismatches when resuming.
        """
        self.__dict__.update(state)

    def __init__(self, env, policy, hidden_size: int = 256):
        """
        Initialize the transformer wrapper.

        Args:
            env: The environment (for observation space info)
            policy: The policy network (must have encode_observations and decode_actions)
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.hidden_size = hidden_size
        self.is_continuous = getattr(policy, "is_continuous", False)
        self._env_memory: Dict[int, Optional[Dict[str, Optional[List[torch.Tensor]]]]] = {}
        self._pending_segment_records: List[SegmentMemoryRecord] = []

        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

    def forward_eval(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference (single timestep).

        For transformers, we still need to handle single-step inference
        but can use the memory from previous sequences.

        Args:
            observations: Observations of shape (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates
        """
        hidden = self.policy.encode_observations(observations, state=state)

        hidden = hidden.unsqueeze(0)

        terminations = state.get("terminations", torch.zeros(1, observations.shape[0], device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0)

        hidden, new_memory = self.policy.transformer(hidden, terminations, state.get("transformer_memory"))

        hidden = hidden.squeeze(0)

        state["transformer_memory"] = self._detach_memory(self._normalize_memory(new_memory))
        state["hidden"] = hidden
        batch_size = hidden.shape[0]
        logits, values = self._decode_actions(hidden, batch_size)

        return logits, values

    def forward(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (handles full BPTT sequences).

        Key difference from LSTM: Transformers process the entire sequence
        at once as context, not step-by-step.

        Args:
            observations: Observations of shape (B, T, ...) or (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates of shape (B, T) or (B,)
        """
        x = observations

        # Determine input shape
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)

        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        x = x.reshape(B * TT, *space_shape)

        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.hidden_size), (
            f"Expected shape ({B * TT}, {self.hidden_size}), got {hidden.shape}"
        )

        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)

        terminations = state.get("terminations", torch.zeros(TT, B, device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0).expand(TT, -1)
        elif terminations.dim() == 2 and terminations.shape[0] == B:
            terminations = terminations.transpose(0, 1)

        hidden, new_memory = self.policy.transformer(hidden, terminations, state.get("transformer_memory"))

        if TT > 1:
            hidden = hidden.transpose(0, 1)
            flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        else:
            hidden = hidden.squeeze(0)
            flat_hidden = hidden

        logits, values = self._decode_actions(flat_hidden, B * TT)

        if TT > 1:
            values = values.reshape(B, TT)

        state["transformer_memory"] = self._detach_memory(self._normalize_memory(new_memory))
        state["hidden"] = hidden

        return logits, values

    def reset_memory(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Initialize memory for a batch of environments.

        This method can be called in two ways:
        1. With explicit batch_size (and optionally device) for initialization
        2. Without arguments (when MettaAgent calls it during reset)

        When called without arguments, we return empty state that will be
        initialized lazily on first forward pass.

        Args:
            batch_size: Optional number of parallel environments
            device: Optional device to create tensors on (ignored, kept for compatibility)

        Returns:
            Initial memory state dictionary
        """
        if batch_size is None:
            self._env_memory.clear()
            self._pending_segment_records = []
            return {
                "transformer_memory": None,
                "hidden": None,
                "needs_init": True,
            }

        if hasattr(self.policy, "initialize_memory"):
            memory = self._normalize_memory(self.policy.initialize_memory(batch_size))
        else:
            memory = None

        self._env_memory.clear()
        self._pending_segment_records = []
        return {
            "transformer_memory": memory,
            "hidden": None,
            "needs_init": False,
        }

    def _detach_memory(self, memory):
        """Detach memory tensors to prevent gradient accumulation.

        Critical for preventing memory leaks during long training runs.
        """
        if memory is None:
            return None
        elif isinstance(memory, torch.Tensor):
            return memory.detach()
        elif isinstance(memory, list):
            return [self._detach_memory(item) for item in memory]
        elif isinstance(memory, tuple):
            return tuple(self._detach_memory(item) for item in memory)
        elif isinstance(memory, dict):
            return {k: self._detach_memory(v) for k, v in memory.items()}
        else:
            return memory

    # ------------------------------------------------------------------
    # Episode memory helpers
    # ------------------------------------------------------------------

    def consume_segment_memory_records(self) -> list[SegmentMemoryRecord]:
        """Return and clear pending segment memory records captured during rollout."""

        records = self._pending_segment_records
        self._pending_segment_records = []
        return records

    def prepare_memory_batch(
        self, snapshots: list[Optional[Dict[str, Optional[List[torch.Tensor]]]]], device: torch.device
    ) -> Optional[Dict[str, Dict[str, List[torch.Tensor]]]]:
        """Convert per-sample memory snapshots into a batch suitable for TransformerModule."""

        if not snapshots or all(snapshot is None for snapshot in snapshots):
            return None

        transformer = getattr(self.policy, "_transformer", None)
        if transformer is None:
            raise AttributeError("Transformer policy must expose '_transformer'")

        mem_len = transformer.memory_len
        if mem_len <= 0:
            return None
        n_layers = transformer.n_layers
        d_model = transformer.d_model

        batched_hidden: List[torch.Tensor] = []

        for layer_idx in range(n_layers):
            layer_entries: List[torch.Tensor] = []
            for snapshot in snapshots:
                layer_tensor = None
                if snapshot is not None and snapshot.get("hidden_states") is not None:
                    hidden_states = snapshot["hidden_states"]
                    if layer_idx < len(hidden_states):
                        layer_tensor = hidden_states[layer_idx]

                if layer_tensor is None or layer_tensor.numel() == 0:
                    layer_entries.append(torch.zeros(mem_len, d_model, device=device))
                    continue

                layer_tensor = layer_tensor.to(device=device)
                if layer_tensor.size(0) < mem_len:
                    pad = torch.zeros(
                        mem_len - layer_tensor.size(0),
                        d_model,
                        device=device,
                        dtype=layer_tensor.dtype,
                    )
                    layer_tensor = torch.cat([pad, layer_tensor], dim=0)
                else:
                    layer_tensor = layer_tensor[-mem_len:]

                layer_entries.append(layer_tensor)

            batched_hidden.append(torch.stack(layer_entries, dim=1))

        return {"transformer_memory": {"hidden_states": batched_hidden}}

    def _build_batch_memory_from_env(
        self, env_indices: torch.Tensor, device: torch.device
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        """Assemble current env memories into a TransformerModule-compatible batch."""

        if not self._env_memory or env_indices.numel() == 0:
            return None

        snapshots: List[Optional[Dict[str, Optional[List[torch.Tensor]]]]] = []
        for env_idx in env_indices.tolist():
            snapshots.append(self._env_memory.get(env_idx))

        prepared = self.prepare_memory_batch(snapshots, device)
        if prepared is None:
            return None
        return prepared["transformer_memory"]

    def _update_env_memory(
        self, env_indices: torch.Tensor, new_memory: Optional[Dict[str, List[torch.Tensor]]]
    ) -> None:
        """Store updated transformer memory per environment after a forward pass."""

        layer_states = self._extract_layer_states(new_memory)
        for batch_pos, env_idx in enumerate(env_indices.tolist()):
            if layer_states is None:
                self._env_memory[env_idx] = None
                continue
            env_layers: List[torch.Tensor] = []
            for layer_hidden in layer_states:
                if layer_hidden.numel() == 0:
                    env_layers.append(layer_hidden.detach().cpu())
                    continue
                if layer_hidden.dim() != 3:
                    raise ValueError(f"Expected layer memory with shape (T, B, D), got {layer_hidden.shape}")
                env_layers.append(layer_hidden[:, batch_pos].detach().cpu())
            self._env_memory[env_idx] = {"hidden_states": env_layers}

    def _record_segment_memory(
        self,
        env_indices: torch.Tensor,
        segment_indices: torch.Tensor,
        segment_positions: torch.Tensor,
        memory_snapshot: Optional[Dict[str, List[torch.Tensor]]],
    ) -> None:
        """Capture memory snapshots for environments beginning a new replay segment."""

        if segment_indices.numel() == 0 or segment_positions.numel() == 0:
            return

        base_layers = self._extract_layer_states(memory_snapshot)
        if base_layers is None:
            snapshots = [None for _ in range(segment_indices.numel())]
        else:
            snapshots = []
            for idx in range(segment_indices.numel()):
                env_layers: List[torch.Tensor] = []
                for layer in base_layers:
                    if layer.numel() == 0:
                        env_layers.append(layer.detach().cpu())
                        continue
                    if layer.dim() != 3:
                        raise ValueError(f"Expected layer memory with shape (T, B, D), got {layer.shape}")
                    env_layers.append(layer[:, idx].detach().cpu())
                snapshots.append({"hidden_states": env_layers})

        for batch_pos, segment_idx in enumerate(segment_indices.tolist()):
            if segment_positions[batch_pos].item() != 0:
                continue
            self._pending_segment_records.append(
                SegmentMemoryRecord(segment_index=segment_idx, memory=snapshots[batch_pos])
            )

    def _decode_actions(self, hidden: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        decode_fn = self.policy.decode_actions
        try:
            return decode_fn(hidden, batch_size=batch_size)
        except TypeError:
            return decode_fn(hidden)

    def _extract_layer_states(
        self, memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]
    ) -> Optional[List[torch.Tensor]]:
        if memory is None:
            return None
        hidden_states = memory.get("hidden_states")
        if hidden_states is None:
            raise ValueError("Transformer memory must include 'hidden_states'")
        if len(hidden_states) == 0:
            return None
        return [layer for layer in hidden_states]

    def _normalize_memory(
        self, memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        if memory is None:
            return None
        hidden_states = memory.get("hidden_states")
        if hidden_states is None:
            raise ValueError("Transformer memory must include 'hidden_states'")
        return {"hidden_states": [layer for layer in hidden_states]}
