"""AGaLiTe transformer component compatible with the PolicyAutoBuilder pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torchrl.data import Composite

from metta.agent.components.agalite_core_enhanced import EnhancedAGaLiTeCore
from metta.agent.components.agalite_kernel import AGaLiTeKernelConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.memory import SegmentMemoryRecord


class AGaLiTeTransformerConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "agalite_transformer"

    hidden_size: int = 192
    n_layers: int = 2
    n_heads: int = 4
    feedforward_size: int = 768
    eta: int = 4
    r: int = 8
    mode: str = "agalite"
    dropout: float = 0.0
    reset_on_terminate: bool = True
    layer_norm_eps: float = 1e-5
    gru_bias: float = 2.0
    kernel: AGaLiTeKernelConfig = Field(default_factory=AGaLiTeKernelConfig)

    def make_component(self, env=None):
        return AGaLiTeTransformer(config=self)


class AGaLiTeTransformer(nn.Module):
    """Sequence model based on the AGaLiTe architecture."""

    def __init__(self, config: AGaLiTeTransformerConfig):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.hidden_size = config.hidden_size

        if self.hidden_size % config.n_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({config.n_heads})")

        self._head_dim = self.hidden_size // config.n_heads
        self._core = EnhancedAGaLiTeCore(
            n_layers=config.n_layers,
            d_model=self.hidden_size,
            d_head=self._head_dim,
            d_ffc=config.feedforward_size,
            n_heads=config.n_heads,
            eta=config.eta,
            r=config.r,
            mode=config.mode,
            reset_on_terminate=config.reset_on_terminate,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            gru_bias=config.gru_bias,
            kernel=config.kernel,
        )

        # Per-environment memory cache used during rollout.
        self._env_memory: Dict[int, Dict[str, Tuple[torch.Tensor, ...]]] = {}
        self._pending_segment_records: List[SegmentMemoryRecord] = []

    # ------------------------------------------------------------------
    # Public nn.Module API
    # ------------------------------------------------------------------
    def forward(self, td: TensorDict) -> TensorDict:
        latent = td[self.in_key]

        total = latent.shape[0]
        bptt = td.get("bptt")
        if bptt is not None:
            time_steps = int(bptt[0].item())
        else:
            time_steps = 1

        batch = total // time_steps
        device = latent.device

        latent = latent.view(time_steps, batch, self.hidden_size)
        terminations = self._compute_terminations(td, time_steps, batch, device)

        segment_indices, segment_pos = self._extract_segment_metadata(td, batch, device)
        snapshots = self._extract_memory_snapshots(td)

        if time_steps == 1:
            env_ids = td.get("training_env_ids")
            if env_ids is None:
                env_ids = torch.arange(batch, device=device)
            env_ids = env_ids.view(batch)

            memory = self._gather_memory(env_ids, device)
            self._record_segment_memory(segment_indices, segment_pos, memory)
            outputs, new_memory = self._core(latent, terminations, memory)
            self._store_memory(env_ids, new_memory)
        else:
            if snapshots is not None:
                memory = self._prepare_memory_batch(snapshots, device)
            else:
                memory = self._core.initialize_memory(batch, device)
            outputs, _ = self._core(latent, terminations, memory)

        if "segment_memory_snapshots" in td.keys():
            del td["segment_memory_snapshots"]
        if "_segment_indices" in td.keys():
            del td["_segment_indices"]
        if "_segment_pos" in td.keys():
            del td["_segment_pos"]

        td[self.out_key] = outputs.reshape(total, self.hidden_size)
        return td

    def initialize_to_environment(self, env, device: torch.device):
        del env
        del device
        self.reset_memory()
        return None

    def reset_memory(self):
        self._env_memory.clear()
        self._pending_segment_records = []

    def get_agent_experience_spec(self) -> Composite:
        # No additional memory needs to be stored in replay; training runs with zero-initialised state.
        return Composite()

    def consume_segment_memory_records(self) -> List[SegmentMemoryRecord]:
        records = self._pending_segment_records
        self._pending_segment_records = []
        return records

    def prepare_memory_batch(
        self, snapshots: List[Optional[Dict[str, Optional[List[torch.Tensor]]]]], device: torch.device
    ) -> Optional[Dict[str, Tuple[torch.Tensor, ...]]]:
        if not snapshots or all(snapshot is None for snapshot in snapshots):
            return None
        return self._prepare_memory_batch(snapshots, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_terminations(
        self,
        td: TensorDict,
        time_steps: int,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        def _reshape(flag_key: str) -> torch.Tensor:
            values = td.get(flag_key)
            if values is None:
                return torch.zeros(time_steps, batch, device=device, dtype=torch.float32)

            vals = values.view(batch, -1)
            if vals.shape[1] != time_steps:
                vals = vals[:, :time_steps]
            vals = vals.float()
            return vals.transpose(0, 1).contiguous()

        dones = _reshape("dones")
        truncateds = _reshape("truncateds")
        terminations = torch.clamp(dones + truncateds, 0.0, 1.0)
        return terminations

    def _gather_memory(self, env_ids: torch.Tensor, device: torch.device):
        aggregated: Dict[str, List[List[torch.Tensor]]] = {}

        for env_id in env_ids.tolist():
            env_memory = self._ensure_env_memory(int(env_id), device)
            for layer_key, tensors in env_memory.items():
                stacked_lists = aggregated.setdefault(layer_key, [list() for _ in range(len(tensors))])
                for idx, tensor in enumerate(tensors):
                    stacked_lists[idx].append(tensor.to(device))

        batch_memory = {
            layer_key: tuple(torch.cat(tensors, dim=0) for tensors in tensor_groups)
            for layer_key, tensor_groups in aggregated.items()
        }

        return batch_memory

    def _store_memory(self, env_ids: torch.Tensor, memory):
        for batch_idx, env_id in enumerate(env_ids.tolist()):
            env_memory = {
                layer_key: tuple(tensor[batch_idx : batch_idx + 1].detach() for tensor in tensors)
                for layer_key, tensors in memory.items()
            }
            self._env_memory[int(env_id)] = env_memory

    def _ensure_env_memory(self, env_id: int, device: torch.device):
        if env_id not in self._env_memory:
            initial_memory = self._core.initialize_memory(1, device)
            env_memory = {key: tuple(tensor.detach() for tensor in tensors) for key, tensors in initial_memory.items()}
            self._env_memory[env_id] = env_memory
        return self._env_memory[env_id]

    def _extract_segment_metadata(
        self, td: TensorDict, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = td.get("_segment_indices")
        positions = td.get("_segment_pos")
        if indices is None or positions is None:
            return (
                torch.zeros(batch, dtype=torch.long, device=device),
                torch.ones(batch, dtype=torch.long, device=device),
            )
        indices = indices.to(device=device, dtype=torch.long).view(-1)
        positions = positions.to(device=device, dtype=torch.long).view(-1)
        return indices, positions

    def _extract_memory_snapshots(
        self, td: TensorDict
    ) -> Optional[List[Optional[Dict[str, Optional[List[torch.Tensor]]]]]]:
        snapshots_data = td.get("segment_memory_snapshots")
        if snapshots_data is None:
            return None

        if isinstance(snapshots_data, NonTensorData):
            data = snapshots_data.data
        else:
            data = snapshots_data

        if data is None:
            return None

        if isinstance(data, list):
            return data

        if isinstance(data, tuple):
            return list(data)

        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            return list(data)

        raise TypeError(f"segment_memory_snapshots must be a sequence of snapshot dicts; received {type(data)!r}")

    def _record_segment_memory(
        self,
        segment_indices: torch.Tensor,
        segment_positions: torch.Tensor,
        memory: Dict[str, Tuple[torch.Tensor, ...]],
    ) -> None:
        if segment_indices.numel() == 0:
            return

        for batch_pos, segment_idx in enumerate(segment_indices.tolist()):
            if segment_positions[batch_pos].item() != 0:
                continue
            snapshot: Dict[str, List[torch.Tensor]] = {}
            for layer_key, tensors in memory.items():
                snapshot[layer_key] = [tensor[batch_pos : batch_pos + 1].detach().cpu() for tensor in tensors]
            self._pending_segment_records.append(SegmentMemoryRecord(segment_index=segment_idx, memory=snapshot))

    def _prepare_memory_batch(
        self,
        snapshots: List[Optional[Dict[str, Optional[List[torch.Tensor]]]]],
        device: torch.device,
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        if not snapshots:
            return self._core.initialize_memory(0, device)

        template = self._core.initialize_memory(1, device)
        batched: Dict[str, List[List[torch.Tensor]]] = {key: [[] for _ in tensors] for key, tensors in template.items()}

        for snapshot in snapshots:
            for layer_key, template_tensors in template.items():
                layer_slices = batched[layer_key]
                if snapshot is None or snapshot.get(layer_key) is None:
                    for idx, tmpl in enumerate(template_tensors):
                        layer_slices[idx].append(torch.zeros_like(tmpl, device=device))
                    continue

                stored_tensors = snapshot[layer_key]
                for idx, tmpl in enumerate(template_tensors):
                    if stored_tensors is None or idx >= len(stored_tensors) or stored_tensors[idx] is None:
                        layer_slices[idx].append(torch.zeros_like(tmpl, device=device))
                        continue
                    tensor = stored_tensors[idx].to(device=device)
                    layer_slices[idx].append(tensor)

        return {
            layer_key: tuple(torch.cat(tensors, dim=0) for tensors in tensor_groups)
            for layer_key, tensor_groups in batched.items()
        }
