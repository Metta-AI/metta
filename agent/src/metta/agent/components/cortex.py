from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from cortex.adapters.metta import MettaTDAdapter
from cortex.stacks import CortexStack
from pydantic import ConfigDict
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig


class CortexTDConfig(ComponentConfig):
    """Config for integrating a Cortex stack via the stateful Metta adapter.

    This component wraps `cortex.adapters.metta.MettaTDAdapter` so it can be
    composed by `PolicyAutoBuilder` like any other Metta component.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_key: str
    out_key: str
    name: str = "cortex"

    d_hidden: int = 128
    out_features: Optional[int] = None

    # Prebuilt Cortex stack instance to use as the core
    stack: CortexStack

    key_prefix: str = "cortex_state"

    def make_component(self, env: Any = None) -> nn.Module:
        return CortexTD(config=self)


class CortexTD(nn.Module):
    """Thin wrapper around the MettaTDAdapter with TD I/O.

    - Delegates TensorDict forward to the internal adapter component
    - Exposes experience spec keys for replay
    - Provides memory helpers for checkpointing
    """

    def __init__(self, config: CortexTDConfig) -> None:
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key

        stack = config.stack
        # Optional sanity check: d_hidden should match the stack's external size
        try:
            stack_hidden = int(stack.cfg.d_hidden)  # type: ignore[attr-defined]
            if stack_hidden != int(config.d_hidden):
                raise ValueError(
                    f"CortexTDConfig.d_hidden ({config.d_hidden}) does not match stack.cfg.d_hidden ({stack_hidden})."
                )
        except Exception:
            # If cfg is not present, skip the check
            pass

        self.core = MettaTDAdapter(
            stack=stack,
            in_key=config.in_key,
            out_key=config.out_key,
            d_hidden=int(config.d_hidden),
            out_features=config.out_features,
            key_prefix=config.key_prefix,
        )

    @torch._dynamo.disable
    def forward(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        return self.core(td)

    def get_agent_experience_spec(self) -> Composite:
        # Advertise minimal keys (training_env_ids) for replay; hidden state is not stored.
        spec_dict: Dict[str, UnboundedDiscrete] = {}
        for key, shape in self.core.experience_keys().items():
            dtype = torch.long if key == "training_env_ids" else torch.float32
            spec_dict[key] = UnboundedDiscrete(shape=torch.Size(shape), dtype=dtype)
        return Composite(spec_dict)

    # Memory passthrough for checkpoint restore
    def get_memory(self):
        return self.core.get_memory()

    def set_memory(self, memory):
        self.core.set_memory(memory)

    def reset_memory(self):
        # No-op by design: trainer/loss call reset_memory() at rollout and
        # train starts. Cortex adapter handles per-step resets internally via
        # (dones | truncateds) masks and maintains caches across minibatches.
        # Clearing here would discard useful context, mirroring transformer
        # policies which keep reset_memory as a no-op.
        pass
