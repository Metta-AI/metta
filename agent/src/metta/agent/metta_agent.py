from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.policy import DistributedPolicy, Policy

logger: Final[logging.Logger] = logging.getLogger("metta.agent.metta_agent")


class MettaAgent(Policy):
    """Compatibility shim for legacy checkpoints that stored MettaAgent objects.

    The modern training stack works directly with ``Policy`` implementations, but
    older checkpoints serialized an instance of ``MettaAgent``. We keep a minimal
    wrapper that delegates to the underlying policy so those checkpoints can still
    be restored.
    """

    # Torch pickling routines call ``__setstate__`` without invoking ``__init__``.
    # We therefore keep the signature permissive and avoid relying on required
    # constructor arguments.
    def __init__(
        self,
        policy: Optional[Policy] = None,
        *,
        device: Union[str, torch.device, None] = None,
        components_with_memory: Optional[list[str]] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.policy: Optional[Policy] = policy
        # ``MettaAgent`` historically tracked which components owned recurrent
        # state. Preserve the attribute so legacy checkpoints remain valid.
        self.components_with_memory: list[str] = list(components_with_memory or [])

        if device is None:
            device = getattr(policy, "device", torch.device("cpu")) if policy else torch.device("cpu")
        self._device: torch.device = torch.device(device)

        # Legacy checkpoints sometimes expect a ``components`` attribute for
        # feature remapping utilities. Ensure it exists even if empty.
        self.components = getattr(policy, "components", nn.ModuleDict())

    # --- Policy interface -------------------------------------------------
    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # type: ignore[override]
        if self.policy is None:
            raise RuntimeError(
                "MettaAgent has no underlying policy; the checkpoint is incompatible with the refactored agent stack."
            )

        if action is None:
            return self.policy(td)

        # Some legacy policies accept ``action`` as a keyword-only argument,
        # while others ignore it altogether. Prefer keyword dispatch to avoid
        # conflicting positional signatures and fall back gracefully.
        try:
            return self.policy(td, action=action)  # type: ignore[misc]
        except TypeError as exc:
            if "action" not in str(exc):
                raise
            return self.policy(td)

    def initialize_to_environment(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if self.policy is None:
            logger.warning("initialize_to_environment called on MettaAgent without a policy")
            return
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(*args, **kwargs)
        self._device = torch.device(kwargs.get("device", getattr(self.policy, "device", self._device)))

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        if self.policy is not None and hasattr(self.policy, "device"):
            return self.policy.device  # type: ignore[return-value]
        return self._device

    def reset_memory(self) -> None:  # type: ignore[override]
        if self.policy is not None and hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def get_agent_experience_spec(self):  # type: ignore[override]
        if self.policy is not None and hasattr(self.policy, "get_agent_experience_spec"):
            return self.policy.get_agent_experience_spec()
        return super().get_agent_experience_spec()

    @property
    def total_params(self) -> int:  # type: ignore[override]
        if self.policy is not None and hasattr(self.policy, "total_params"):
            return self.policy.total_params  # type: ignore[return-value]
        return super().total_params

    # ------------------------------------------------------------------
    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        super().to(*args, **kwargs)
        if self.policy is not None and hasattr(self.policy, "to"):
            self.policy = self.policy.to(*args, **kwargs)  # type: ignore[assignment]
        if args:
            maybe_device = args[0]
        else:
            maybe_device = kwargs.get("device")
        if maybe_device is not None:
            self._device = torch.device(maybe_device)
        return self

    def __getattr__(self, name: str):
        # Delegate to underlying policy for compatibility with legacy code paths
        if name in {"policy", "_device", "components", "components_with_memory"}:
            raise AttributeError(name)
        policy = object.__getattribute__(self, "policy")
        if policy is not None and hasattr(policy, name):
            return getattr(policy, name)
        raise AttributeError(name)


class DistributedMettaAgent(DistributedPolicy):
    """DistributedDataParallel wrapper used by legacy checkpoints."""

    def __init__(self, agent: Policy | MettaAgent, device: torch.device) -> None:
        # Unwrap DistributedPolicy instances to avoid nesting
        if isinstance(agent, DistributedPolicy):
            base_policy = agent.module
        else:
            base_policy = agent
        super().__init__(base_policy, device)


if TYPE_CHECKING:
    from typing import TypeAlias

    PolicyAgent: TypeAlias = Union[Policy, MettaAgent, DistributedPolicy]
else:
    PolicyAgent = Union[Policy, MettaAgent, DistributedPolicy]


__all__ = ["MettaAgent", "DistributedMettaAgent", "PolicyAgent"]
