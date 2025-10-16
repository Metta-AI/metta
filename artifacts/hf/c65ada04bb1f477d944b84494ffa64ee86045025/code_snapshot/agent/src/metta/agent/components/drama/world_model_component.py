from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite

from metta.rl.training import EnvironmentMetaData

from .config import DramaWorldModelConfig
from .mamba_wrapper import MambaConfig as _DramaMambaConfig
from .mamba_wrapper import MambaWrapperModel


class DramaWorldModelComponent(nn.Module):
    """Adapter around DRAMA's Mamba world-model backbone."""

    def __init__(self, config: DramaWorldModelConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.action_key = config.action_key
        self.pool: Literal["cls", "mean", "none"] = "mean"

        action_dim = max(1, config.action_dim)
        if env is not None:
            env_action_space = getattr(env, "action_space", None)
            resolved_action_dim = getattr(env_action_space, "n", None)
            if resolved_action_dim is not None:
                action_dim = max(1, int(resolved_action_dim))

        self.action_dim = action_dim
        self.config.action_dim = action_dim

        mamba_cfg = _DramaMambaConfig(
            d_model=config.d_model,
            d_intermediate=config.d_intermediate,
            n_layer=config.n_layer,
            stoch_dim=config.stoch_dim,
            action_dim=self.action_dim,
            dropout_p=config.dropout_p,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            pff_cfg=config.pff_cfg,
        )
        self.backbone = MambaWrapperModel(mamba_cfg)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, td: TensorDict) -> TensorDict:
        samples = td[self.in_key]
        actions = td.get(self.action_key)

        if samples.dim() == 2:
            samples = samples.unsqueeze(1)

        batch_size, seq_len = samples.shape[0], samples.shape[1]

        if actions is None:
            actions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=samples.device)
        else:
            actions = actions.to(device=samples.device)
            # Collapse one-hot or singleton action dimensions to scalars.
            if actions.dim() > 1:
                if actions.size(-1) > 1:
                    actions = actions.argmax(dim=-1)
                else:
                    actions = actions.squeeze(-1)

            actions = actions.reshape(batch_size, -1)

            if actions.size(1) == 1:
                actions = actions.expand(-1, seq_len)
            elif actions.size(1) < seq_len:
                pad = torch.zeros(
                    (batch_size, seq_len - actions.size(1)),
                    dtype=actions.dtype,
                    device=samples.device,
                )
                actions = torch.cat([actions, pad], dim=1)
            elif actions.size(1) > seq_len:
                actions = actions[:, :seq_len]

        hidden = self.backbone(samples, actions.long())
        hidden = self.output_norm(hidden)
        pooled = hidden.mean(dim=1)
        td.set(self.out_key, pooled)
        return td

    def get_agent_experience_spec(self) -> Composite:
        return Composite({})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        self.to(device)
        return None

    def reset_memory(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
