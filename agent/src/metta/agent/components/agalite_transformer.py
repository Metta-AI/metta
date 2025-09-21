"""AGaLiTe transformer component compatible with the PolicyAutoBuilder pipeline."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite

from metta.agent.components.agalite_core_enhanced import EnhancedAGaLiTeCore
from metta.agent.components.component_config import ComponentConfig


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
        )

        # Per-environment memory cache used during rollout.
        self._env_memory: Dict[int, Dict[str, Tuple[torch.Tensor, ...]]] = {}

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

        if time_steps == 1:
            env_ids = td.get("training_env_ids")
            if env_ids is None:
                env_ids = torch.arange(batch, device=device)
            env_ids = env_ids.view(batch)

            memory = self._gather_memory(env_ids, device)
            outputs, new_memory = self._core(latent, terminations, memory)
            self._store_memory(env_ids, new_memory)
        else:
            memory = self._core.initialize_memory(batch, device)
            outputs, _ = self._core(latent, terminations, memory)

        td[self.out_key] = outputs.reshape(total, self.hidden_size)
        return td

    def initialize_to_environment(self, env, device: torch.device):
        del env
        del device
        self.reset_memory()
        return None

    def reset_memory(self):
        self._env_memory.clear()

    def get_agent_experience_spec(self) -> Composite:
        # No additional memory needs to be stored in replay; training runs with zero-initialised state.
        return Composite()

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
