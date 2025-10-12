from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite

from metta.agent.components.utils import zero_long
from metta.rl.training import EnvironmentMetaData

from .config import MambaBackboneConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .wrapper import MambaConfig as _WrapperConfigType
    from .wrapper import MambaWrapperModel as _WrapperModelType
else:
    _WrapperConfigType = Any
    _WrapperModelType = Any

_WRAPPER_TYPES: Optional[Tuple[type, type]] = None


def _load_wrapper_types() -> Tuple[type, type]:
    global _WRAPPER_TYPES
    if _WRAPPER_TYPES is not None:
        return _WRAPPER_TYPES
    try:
        from .wrapper import MambaConfig as wrapper_config
        from .wrapper import MambaWrapperModel as wrapper_model
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm":
            raise RuntimeError(
                "MambaBackboneComponent requires the `mamba-ssm` package; install it on a CUDA-enabled Linux host "
                "or disable Mamba-based recipes."
            ) from exc
        raise
    _WRAPPER_TYPES = (wrapper_config, wrapper_model)
    return _WRAPPER_TYPES


@dataclass
class _EnvState:
    inference_params: "_CacheWrapper"
    position: int = 0

    def reset(self) -> None:
        self.inference_params.reset()
        self.position = 0


class _CacheWrapper:
    def __init__(self, cache_dict: dict, max_seqlen: int):
        self.key_value_memory_dict = cache_dict
        self.seqlen_offset = 0
        self.max_seqlen = max_seqlen

    def reset(self) -> None:
        for cache in self.key_value_memory_dict.values():
            if torch.is_tensor(cache):
                cache.zero_()
            elif isinstance(cache, dict):
                for value in cache.values():
                    if torch.is_tensor(value):
                        value.zero_()
        self.seqlen_offset = 0


class MambaBackboneComponent(nn.Module):
    """Streaming-friendly Mamba backbone matching the policy wrapper contract."""

    def __init__(self, config: MambaBackboneConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.pool: Literal["cls", "mean", "none"] = config.pool
        self.use_aux_tokens = config.use_aux_tokens
        self.last_action_dim = max(1, config.last_action_dim)
        self.max_cache_size = max(1, config.max_cache_size)
        wrapper_config_cls, wrapper_model_cls = _load_wrapper_types()
        self._wrapper_config_type: type = wrapper_config_cls
        self._wrapper_model_cls: type = wrapper_model_cls

        self._resolved_ssm_cfg = self.config.resolved_ssm_cfg()
        self.wrapper = self._wrapper_model_cls(self._build_wrapper_config(self._resolved_ssm_cfg))
        self._mem_eff_enabled = bool(self._resolved_ssm_cfg.get("use_mem_eff_path", True))

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        if self.use_aux_tokens:
            self.reward_proj = nn.Linear(1, config.d_model)
            self.reset_proj = nn.Linear(1, config.d_model)
            self.action_proj = nn.Linear(self.last_action_dim, config.d_model)
        else:
            self.reward_proj = None
            self.reset_proj = None
            self.action_proj = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_p)

        self._env_states: Dict[int, _EnvState] = {}
        self._init_parameters()

    def _build_wrapper_config(self, ssm_cfg: dict[str, object]) -> _WrapperConfigType:
        return self._wrapper_config_type(
            d_model=self.config.d_model,
            d_intermediate=self.config.d_intermediate,
            n_layer=self.config.n_layer,
            stoch_dim=self.config.d_model,
            action_dim=self.last_action_dim,
            dropout_p=self.config.dropout_p,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=self.config.attn_layer_idx,
            attn_cfg=self.config.attn_cfg,
        )

    def _rebuild_wrapper(self, *, enable_mem_eff: Optional[bool] = None) -> None:
        ssm_cfg = dict(self.config.resolved_ssm_cfg())
        if enable_mem_eff is not None:
            ssm_cfg["use_mem_eff_path"] = enable_mem_eff
        self._resolved_ssm_cfg = ssm_cfg
        self.wrapper = self._wrapper_model_cls(self._build_wrapper_config(self._resolved_ssm_cfg))
        self._mem_eff_enabled = bool(self._resolved_ssm_cfg.get("use_mem_eff_path", True))
        self.reset_memory()

    def _supports_mem_eff_fallback(self) -> bool:
        return bool(self.config.auto_align_stride)

    def _init_parameters(self) -> None:
        """Initialize projection layers for stable SSM training."""

        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        if self.use_aux_tokens:
            for proj in (self.reward_proj, self.reset_proj, self.action_proj):
                if proj is not None:
                    nn.init.xavier_uniform_(proj.weight)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        nn.init.normal_(self.cls_token, mean=0.0, std=1.0 / self.config.d_model**0.5)

    def _handle_wrapper_runtime(
        self,
        exc: RuntimeError,
        samples: torch.Tensor,
        action: torch.Tensor,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if "causal_conv1d" not in str(exc) or not self._mem_eff_enabled or not self._supports_mem_eff_fallback():
            raise
        logger.warning(
            "Disabling Mamba mem-efficient path after causal conv failure: %s",
            exc,
        )
        self._rebuild_wrapper(enable_mem_eff=False)
        return self.wrapper(samples, action, inference_params=inference_params, **kwargs)

    def _run_wrapper(self, samples: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        try:
            return self.wrapper(samples, action, **kwargs)
        except RuntimeError as exc:
            return self._handle_wrapper_runtime(exc, samples, action, **kwargs)

    # ------------------------------------------------------------------
    # Token preparation
    # ------------------------------------------------------------------
    def _build_tokens(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tokens, reset_flags) prepared for the Mamba wrapper.

        tokens has shape [batch, time, num_tokens, d_model] with sequence
        [CLS token, observation embeddings..., reward+reset token, action token].
        reset_flags has shape [batch, time] marking episode boundaries (done|truncated).
        """

        x = td[self.in_key]
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        x = self.input_proj(x)

        batch_flat = td.batch_size.numel()
        tt = int(td.get("bptt", torch.ones(1, device=device))[0].item())
        batch = max(batch_flat // tt, 1)

        x = rearrange(x, "(b tt) s d -> b tt s d", b=batch, tt=tt)

        zeros = torch.zeros(batch_flat, device=device)
        rewards = td.get("rewards", zeros)
        dones = td.get("dones", zeros)
        truncateds = td.get("truncateds", zeros)
        reset_flags = torch.logical_or(dones.bool(), truncateds.bool()).reshape(batch, tt)

        if self.use_aux_tokens:
            zeros = torch.zeros(batch_flat, device=device)
            rewards = rearrange(rewards, "(b tt) -> b tt 1 1", b=batch, tt=tt).float()
            reward_token = self.reward_proj(rewards)

            resets = reset_flags.float().reshape(batch, tt, 1, 1)
            reset_token = self.reset_proj(resets)

            last_actions = td.get("last_actions")
            if last_actions is None:
                last_actions = torch.zeros(batch_flat, self.last_action_dim, device=device)
            else:
                last_actions = last_actions.to(device=device)
                if last_actions.dim() == 1:
                    last_actions = last_actions.unsqueeze(-1)
                last_actions = last_actions.reshape(batch_flat, -1)[..., : self.last_action_dim]

            last_actions = rearrange(last_actions, "(b tt) d -> b tt 1 d", b=batch, tt=tt)
            action_token = self.action_proj(last_actions.float())
        else:
            reward_token = torch.zeros_like(x[..., :1, :])
            reset_token = torch.zeros_like(x[..., :1, :])
            action_token = torch.zeros_like(x[..., :1, :])

        cls = self.cls_token.to(device).expand(x.size(0), x.size(1), -1, -1)
        tokens = torch.cat([cls, x, reward_token + reset_token, action_token], dim=2)
        return tokens, reset_flags

    def _dummy_actions(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return zero_long((batch, seq_len), device=device)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return hidden[:, 0, :]
        if self.pool == "mean":
            return hidden.mean(dim=1)
        if self.pool == "none":
            b, s, d = hidden.shape
            return hidden.reshape(b, s * d)
        raise ValueError(f"Unsupported pool mode: {self.pool}")

    def _ensure_state(self, env_id: int, tokens_per_step: int, device: torch.device, dtype: torch.dtype) -> _EnvState:
        state = self._env_states.get(env_id)
        if state is not None and state.inference_params.max_seqlen >= tokens_per_step:
            return state

        cache = self.wrapper.allocate_inference_cache(
            batch_size=1,
            max_seqlen=self.max_cache_size * tokens_per_step,
            dtype=dtype,
        )
        cache_wrapper = _CacheWrapper(cache, self.max_cache_size * tokens_per_step)
        state = _EnvState(cache_wrapper)
        self._env_states[env_id] = state
        return state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, td: TensorDict) -> TensorDict:
        tokens, reset_flags = self._build_tokens(td)
        device = tokens.device
        dtype = tokens.dtype

        tt = tokens.size(1)
        batch = tokens.size(0)
        tokens_per_step = tokens.size(2)

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            env_ids = torch.arange(batch, device=device, dtype=torch.long)
        else:
            env_ids = training_env_ids.reshape(-1).to(device=device, dtype=torch.long)

        use_streaming_path = not self.training and tt == 1

        if use_streaming_path:
            outputs = []
            positions = torch.zeros(batch, dtype=torch.long, device=device)
            resets_step = (
                reset_flags[:, 0] if reset_flags.numel() else torch.zeros(batch, dtype=torch.bool, device=device)
            )
            tokens = tokens.reshape(batch, tokens_per_step, -1)
            dummy_action = self._dummy_actions(1, 1, device)
            for idx in range(batch):
                env_id = int(env_ids[idx].item())
                state = self._ensure_state(env_id, tokens_per_step, device, dtype)

                cache = state.inference_params
                cache.seqlen_offset = state.position

                hidden_steps = []
                for token_idx in range(tokens_per_step):
                    token = tokens[idx : idx + 1, token_idx : token_idx + 1]
                    hidden = self._run_wrapper(
                        token,
                        dummy_action,
                        inference_params=cache,
                        num_last_tokens=1,
                    )
                    hidden_steps.append(hidden[:, -1:])
                    state.position = min(state.position + 1, cache.max_seqlen)
                    cache.seqlen_offset = state.position

                hidden = torch.cat(hidden_steps, dim=1)
                hidden = self.norm(self.dropout(hidden))
                pooled = self._pool(hidden)
                outputs.append(pooled.squeeze(0))

                if resets_step[idx]:
                    state.reset()
                positions[idx] = state.position

            td.set(self.out_key, torch.stack(outputs, dim=0))
            td.set("transformer_position", positions)
            return td

        # Batched path (training or evaluation with longer sequences)
        for env_id in env_ids.detach().cpu().tolist():
            state = self._env_states.get(int(env_id))
            if state is not None:
                state.reset()

        seq = tokens.reshape(batch * tt, tokens_per_step, -1)
        hidden = self._run_wrapper(
            seq,
            self._dummy_actions(batch * tt, tokens_per_step, device),
            inference_params=None,
        )
        hidden = self.norm(self.dropout(hidden))
        pooled = self._pool(hidden)
        td.set(self.out_key, pooled)

        return td

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------
    def get_agent_experience_spec(self) -> Composite:
        return Composite({})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        self._env_states.clear()
        return None

    def reset_memory(self) -> None:
        for state in self._env_states.values():
            state.reset()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
