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

        memory = state.get("transformer_memory", None)

        hidden = hidden.unsqueeze(0)

        terminations = state.get("terminations", torch.zeros(1, observations.shape[0], device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0)

        hidden, new_memory = self.policy.transformer(hidden, terminations, memory)

        hidden = hidden.squeeze(0)

        if new_memory is not None:
            state["transformer_memory"] = self._detach_memory(new_memory)
        else:
            state["transformer_memory"] = new_memory
        state["hidden"] = hidden
        logits, values = self.policy.decode_actions(hidden)

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
        memory = state.get("transformer_memory", None)

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

        hidden, new_memory = self.policy.transformer(hidden, terminations, memory)

        if TT > 1:
            hidden = hidden.transpose(0, 1)
            flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        else:
            hidden = hidden.squeeze(0)
            flat_hidden = hidden

        logits, values = self.policy.decode_actions(flat_hidden)

        if TT > 1:
            values = values.reshape(B, TT)

        if new_memory is not None:
            state["transformer_memory"] = self._detach_memory(new_memory)
        else:
            state["transformer_memory"] = new_memory
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
            memory = self.policy.initialize_memory(batch_size)
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
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        """Convert per-sample memory snapshots into a batch suitable for TransformerModule."""

        if not snapshots or all(snapshot is None for snapshot in snapshots):
            return None

        transformer = getattr(self.policy, "_transformer", None)
        if transformer is None or transformer.memory_len <= 0:
            return None

        n_layers = transformer.n_layers
        mem_len = transformer.memory_len
        d_model = transformer.d_model
        batched_hidden: List[torch.Tensor] = []

        for layer_idx in range(n_layers):
            layer_entries: List[torch.Tensor] = []
            for snapshot in snapshots:
                if snapshot is None or snapshot.get("hidden_states") is None:
                    layer_entries.append(torch.zeros(mem_len, d_model, device=device))
                    continue
                layer_tensor = snapshot["hidden_states"][layer_idx]
                if layer_tensor.numel() == 0:
                    layer_entries.append(torch.zeros(mem_len, d_model, device=device))
                    continue
                layer_tensor = layer_tensor.to(device=device)
                if layer_tensor.size(0) < mem_len:
                    pad = torch.zeros(mem_len - layer_tensor.size(0), d_model, device=device, dtype=layer_tensor.dtype)
                    layer_tensor = torch.cat([pad, layer_tensor], dim=0)
                layer_entries.append(layer_tensor)
            batched_hidden.append(torch.stack(layer_entries, dim=1))

        return {"hidden_states": batched_hidden}

    def _build_batch_memory_from_env(
        self, env_indices: torch.Tensor, device: torch.device
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        """Assemble current env memories into a TransformerModule-compatible batch."""

        if not self._env_memory or env_indices.numel() == 0:
            return None

        snapshots: List[Optional[Dict[str, Optional[List[torch.Tensor]]]]] = []
        for env_idx in env_indices.tolist():
            snapshots.append(self._env_memory.get(env_idx))

        return self.prepare_memory_batch(snapshots, device)

    def _update_env_memory(self, env_indices: torch.Tensor, new_memory: Dict[str, List[torch.Tensor]]) -> None:
        """Store updated transformer memory per environment after a forward pass."""

        if new_memory is None or "hidden_states" not in new_memory:
            return

        for batch_pos, env_idx in enumerate(env_indices.tolist()):
            if new_memory["hidden_states"] is None:
                self._env_memory[env_idx] = None
                continue
            layer_snapshots: List[torch.Tensor] = []
            for layer_hidden in new_memory["hidden_states"]:
                layer_snapshots.append(layer_hidden[:, batch_pos].detach().cpu())
            self._env_memory[env_idx] = {"hidden_states": layer_snapshots}

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

        if memory_snapshot is None or memory_snapshot.get("hidden_states") is None:
            snapshots = [None for _ in range(segment_indices.numel())]
        else:
            snapshots = [
                {"hidden_states": [layer[:, idx].detach().cpu() for layer in memory_snapshot["hidden_states"]]}
                for idx in range(segment_indices.numel())
            ]

        for batch_pos, segment_idx in enumerate(segment_indices.tolist()):
            if segment_positions[batch_pos].item() != 0:
                continue
            self._pending_segment_records.append(
                SegmentMemoryRecord(segment_index=segment_idx, memory=snapshots[batch_pos])
            )
