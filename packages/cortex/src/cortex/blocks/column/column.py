"""Column: global prior gating with optional per-token refinement, E-axis mixer, and outer ReZero."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch._dynamo import disable

from cortex.blocks.base import BaseBlock
from cortex.blocks.column.routers import GlobalContextRouter, TokenRefiner
from cortex.blocks.registry import build_block, register_block
from cortex.cells import build_cell
from cortex.cells.base import MemoryCell
from cortex.config import BlockConfig, ColumnBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(ColumnBlockConfig)
class ColumnBlock(BaseBlock):
    """Mix multiple experts using a router that outputs a global gate."""

    def __init__(self, config: ColumnBlockConfig, d_hidden: int, cell: MemoryCell | None = None) -> None:
        super().__init__(d_hidden=d_hidden, cell=self._make_placeholder_cell(d_hidden))
        self.config = config
        self.d_hidden = d_hidden
        self.experts = nn.ModuleList(self._build_experts(config.experts, d_hidden))
        num_experts = len(self.experts)
        # Keep boundary norm regardless of expert count.
        self.boundary_norm = nn.RMSNorm(d_hidden)

        # Initialize router/refiner/mixer only when there is actual expert mixing.
        if num_experts > 1:
            self.router: GlobalContextRouter | None = GlobalContextRouter(d_hidden, num_experts, config.router)
            lam0 = float(getattr(config.router, "whisper_lambda", 0.0))
            self.refiner: TokenRefiner | None = (
                TokenRefiner(d_hidden, num_experts, config.router) if lam0 > 0.0 else None
            )
            d_k_mix = int(config.router.d_key or d_hidden)
            self.e_mixer: _EAxisCrossAttention | None = _EAxisCrossAttention(
                d_hidden=d_hidden, num_experts=num_experts, d_key=d_k_mix
            )
        else:
            # Avoid allocating unused parameters when E=1; fast path in forward will bypass them.
            self.router = None
            self.refiner = None
            self.e_mixer = None

        self.head = _ColumnReZeroHead(d_hidden=d_hidden, hidden_mult=2)
        init_alpha = float(getattr(config, "alpha_init", 0.01))
        self.alpha_col = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.alpha_main = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        # Precompute stable per‑expert state keys once.
        self._expert_keys: list[str] = [self._expert_state_key(i, expert) for i, expert in enumerate(self.experts)]
        self._compiled_experts: list | None = None
        # Lazily created CUDA streams for expert parallelism (one per expert).
        self._cuda_streams: list[torch.cuda.Stream] | None = None

    @staticmethod
    def _make_placeholder_cell(hidden_size: int) -> MemoryCell:
        """Minimal placeholder MemoryCell for BaseBlock."""

        class _NoOpCell(MemoryCell):
            def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
                return TensorDict({}, batch_size=[batch], device=torch.device(device))

            def forward(self, x: Tensor, state: MaybeState, *, resets: Optional[ResetMask] = None):
                return x, state

            def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
                return state

        return _NoOpCell(hidden_size)

    def _build_experts(self, expert_cfgs: list[BlockConfig], d_hidden: int) -> list[BaseBlock]:
        experts: list[BaseBlock] = []
        for cfg in expert_cfgs:
            if cfg.cell is None:
                block = build_block(config=cfg, d_hidden=d_hidden, cell=None)  # type: ignore[arg-type]
            else:
                hs = cfg.get_cell_hidden_size(d_hidden)
                dumped = cfg.cell.model_dump()
                dumped["hidden_size"] = hs
                cell_config = type(cfg.cell)(**dumped)
                cell = build_cell(cell_config)
                block = build_block(config=cfg, d_hidden=d_hidden, cell=cell)
            experts.append(block)
        return experts

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        # Build in one shot to avoid per‑item __setitem__ during graph capture.
        state_map = {
            key: expert.init_state(batch=batch, device=device, dtype=dtype)
            for key, expert in zip(self._expert_keys, self.experts, strict=False)
        }
        return TensorDict(state_map, batch_size=[batch], device=torch.device(device))

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None or not isinstance(state, TensorDict):
            return state
        batch_size = state.batch_size[0] if state.batch_size else mask.shape[0]
        # Build mapping in Python first, then wrap.
        state_map = {}
        for key, expert in zip(self._expert_keys, self.experts, strict=False):
            cur = state.get(key)
            state_map[key] = expert.reset_state(cur, mask)
        return TensorDict(state_map, batch_size=[batch_size], device=state.device)

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        make_td = disable(
            lambda state_map, batch_size, device: TensorDict(
                state_map, batch_size=[batch_size], device=torch.device(device)
            )
        )
        make_empty_td = disable(
            lambda batch_size, device: TensorDict({}, batch_size=[batch_size], device=torch.device(device))
        )
        is_step = x.dim() == 2
        B = x.shape[0]
        td_empty = make_empty_td(B, x.device)
        # Boundary normalization
        u = self.boundary_norm(x)
        use_compiled = self._compiled_experts is not None and torch.is_grad_enabled()
        expert_call_list = self._compiled_experts if use_compiled else list(self.experts)

        num_experts = len(expert_call_list)
        # Preallocate lists for deterministic ordering (avoid shared refs)
        expert_outs: list[Tensor | None] = [None] * num_experts
        expert_states: list[TensorDict | None] = [None] * num_experts

        # Parallelize with CUDA streams unless disabled or under graph capture
        can_parallel = num_experts > 1 and x.is_cuda and torch.cuda.is_available()

        if can_parallel:
            # Lazily create one stream per expert on the input tensor's device.
            dev = x.device
            if (
                self._cuda_streams is None
                or len(self._cuda_streams) != num_experts
                or any(s.device != dev for s in (self._cuda_streams or []))
            ):
                self._cuda_streams = [torch.cuda.Stream(device=dev) for _ in range(num_experts)]

            current = torch.cuda.current_stream(dev)
            # Schedule each expert on its own stream.
            for i, (key, expert) in enumerate(zip(self._expert_keys, expert_call_list, strict=False)):
                s = self._cuda_streams[i]
                # Ensure expert stream sees work done on current stream for inputs.
                s.wait_stream(current)
                with torch.cuda.stream(s):
                    expert_state = state.get(key) if isinstance(state, TensorDict) else td_empty
                    if expert_state is None:
                        expert_state = td_empty
                    y_i, s_i = expert(u, expert_state, resets=resets)
                    expert_outs[i] = y_i
                    expert_states[i] = s_i if isinstance(s_i, TensorDict) else make_empty_td(B, x.device)

            # Back on current stream, wait for all expert streams before consuming outputs.
            for s in self._cuda_streams:
                current.wait_stream(s)
        else:
            # Fallback sequential execution (CPU or single expert)
            for i, (key, expert) in enumerate(zip(self._expert_keys, expert_call_list, strict=False)):
                expert_state = state.get(key) if isinstance(state, TensorDict) else td_empty
                if expert_state is None:
                    expert_state = td_empty
                y_i, s_i = expert(u, expert_state, resets=resets)
                expert_outs[i] = y_i
                expert_states[i] = s_i if isinstance(s_i, TensorDict) else make_empty_td(B, x.device)

        # Build the next state TensorDict on the current stream after synchronization.
        # Replace None placeholders and build state map
        fixed_states = [v if isinstance(v, TensorDict) else make_empty_td(B, x.device) for v in expert_states]
        state_map = {k: v for k, v in zip(self._expert_keys, fixed_states, strict=False)}
        next_state = make_td(state_map, B, x.device)

        # Ensure expert_outs are all set
        expert_outs_tensors: list[Tensor] = [
            (y if isinstance(y, torch.Tensor) else torch.empty(0, device=x.device)) for y in expert_outs
        ]

        # Fast‑path for a single expert: keep RMSNorm + ReZero semantics but
        # bypass E‑axis mixer and router/refiner overhead. Algebraically,
        # with E=1 the mixture reduces to y_minus_x = (y - u) + (u - x) = (y - x).
        if len(expert_outs_tensors) == 1:
            y_single = expert_outs_tensors[0]
            y_minus_x = y_single - x  # [B,T,H]
            y_total = x + self.alpha_main.to(y_minus_x.dtype) * y_minus_x
            h = self.head(y_minus_x)
            out = y_total + self.alpha_col.to(h.dtype) * h
            return out, next_state

        # Build deltas across experts while keeping a token dimension in both modes.
        x_tokens = x.unsqueeze(1) if is_step else x  # [B,T,H]
        U_tokens = u.unsqueeze(1) if is_step else u  # [B,T,H]
        expert_tokens = [y_i.unsqueeze(1) if is_step else y_i for y_i in expert_outs_tensors]
        D_list = [(y_tok - U_tokens) for y_tok in expert_tokens]
        D = torch.stack(D_list, dim=2)  # [B,T,E,H]

        # E-axis mixing (cross-attention residual)
        D_mixed = self.e_mixer(U_tokens, D)  # [B,T,E,H]

        restrict = bool(getattr(self.config.router, "restrict_to_topk", True))
        g_logits = self.router.global_logits(restrict_topk=restrict).to(D_mixed.dtype)  # [E]

        lam = float(getattr(self.config.router, "whisper_lambda", 0.0))
        if lam > 0.0:
            p_t = self.refiner(U_tokens)  # type: ignore[operator]  # [B,T,E]
            logits_total = g_logits.view(1, 1, -1) + (lam * p_t)
            a_t = torch.softmax(logits_total.to(dtype=torch.float32), dim=-1).to(D_mixed.dtype)
        else:
            # Broadcast global prior over tokens
            a_t = torch.softmax(g_logits.to(dtype=torch.float32), dim=-1).to(D_mixed.dtype)
            a_t = a_t.view(1, 1, -1).expand(D_mixed.shape[0], D_mixed.shape[1], -1)

        # Reduce across experts to form the total mixture and then apply ReZero around it:
        # y_total = sum_k a_{t,k} y_k = x + [sum_k a_{t,k} (y_k - u) + (u - x)]
        y_delta = (a_t.unsqueeze(-1) * D_mixed).sum(dim=2)  # [B,T,H]
        res_corr = U_tokens - x_tokens  # [B,T,H]
        y_minus_x = y_delta + res_corr  # equals y_total - x
        y_total = x_tokens + self.alpha_main.to(y_minus_x.dtype) * y_minus_x
        h = self.head(y_minus_x)  # small corrective head on the residual
        out = y_total + self.alpha_col.to(h.dtype) * h
        return (out.squeeze(1) if is_step else out), next_state

    @staticmethod
    def _expert_state_key(i: int, expert: nn.Module) -> str:
        cls = expert.__class__.__name__
        return f"expert_{cls}_{i}"


__all__ = ["ColumnBlock"]


class _EAxisCrossAttention(nn.Module):
    """Cross-attention mixer over experts (E-axis) with residual connection."""

    def __init__(self, d_hidden: int, num_experts: int, d_key: int | None = None) -> None:
        super().__init__()
        d_k = int(d_key or d_hidden)
        self.d_k = d_k
        self.num_experts = int(num_experts)
        self.Wq = nn.Linear(d_hidden, d_k, bias=False)
        self.Wk = nn.Linear(d_hidden, d_k, bias=False)
        self.Wv = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_hidden, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(num_experts, d_k))

        # Near-identity init: small weights and zero out-proj
        for lin in (self.Wq, self.Wk, self.Wv):
            nn.init.uniform_(lin.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.out.weight)
        nn.init.uniform_(self.q_bias, a=-1e-3, b=1e-3)

    def forward(self, u_tokens: Tensor, deltas: Tensor) -> Tensor:
        """u_tokens: [B,T,H]; deltas: [B,T,E,H] -> [B,T,E,H]."""
        assert u_tokens.dim() == 3 and deltas.dim() == 4
        B, T, E, H = deltas.shape
        # Build Q for each expert by broadcasting u_t and adding per-expert bias
        Qt = self.Wq(u_tokens)  # [B,T,d_k]
        Q = Qt.unsqueeze(2) + self.q_bias.view(1, 1, E, -1)  # [B,T,E,d_k]
        K = self.Wk(deltas)  # [B,T,E,d_k]
        V = self.Wv(deltas)  # [B,T,E,H]

        # Flatten tokens to batch compute attention over E
        N = B * T
        Qn = Q.view(N, E, self.d_k)
        Kn = K.view(N, E, self.d_k)
        Vn = V.view(N, E, H)
        attn_scores = torch.matmul(Qn, Kn.transpose(1, 2)) / math.sqrt(self.d_k)  # [N,E,E]
        attn = torch.softmax(attn_scores.to(dtype=torch.float32), dim=-1).to(Vn.dtype)
        On = torch.matmul(attn, Vn)  # [N,E,H]
        out_tokens = On.view(B, T, E, H)
        return deltas + self.out(out_tokens)


class _ColumnReZeroHead(nn.Module):
    """Tiny FFN with zeroed last layer for calm starts."""

    def __init__(self, d_hidden: int, hidden_mult: int = 2) -> None:
        super().__init__()
        d_mid = max(1, int(hidden_mult) * d_hidden)
        self.fc1 = nn.Linear(d_hidden, d_mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_mid, d_hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))
