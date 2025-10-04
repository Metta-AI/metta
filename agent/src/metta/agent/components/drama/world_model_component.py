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

        mamba_cfg = _DramaMambaConfig(
            d_model=config.d_model,
            d_intermediate=config.d_intermediate,
            n_layer=config.n_layer,
            stoch_dim=config.stoch_dim,
            action_dim=config.action_dim,
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
        if actions is None:
            actions = torch.zeros(samples.size(0), samples.size(1), dtype=torch.long, device=samples.device)
        if actions.dim() == 2 and actions.size(-1) > 1:
            actions = actions.argmax(dim=-1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

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
