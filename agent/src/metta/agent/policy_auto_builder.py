import logging
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential
from torch.nn.parameter import UninitializedParameter
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.training import GameRules
from mettagrid.base_config import Config

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


@dataclass
class _ActivationTracker:
    """Track ReLU activation rates per feature."""

    total: torch.Tensor
    active: torch.Tensor

    @classmethod
    def new(cls, features: int) -> "_ActivationTracker":
        zeros = torch.zeros(features, dtype=torch.float64)
        return cls(total=zeros.clone(), active=zeros.clone())

    def update(self, output: torch.Tensor) -> None:
        tensor = output.detach()
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            flat = tensor.unsqueeze(0)
        else:
            last_dim = tensor.shape[-1]
            flat = tensor.reshape(-1, last_dim)

        active_counts = (flat > 0).sum(dim=0, dtype=torch.float64)
        active_counts = active_counts.to(self.active.device)
        batch = float(flat.shape[0])

        if self.total.numel() != active_counts.numel():
            raise ValueError("Activation tracker feature size mismatch.")

        self.total += batch
        self.active += active_counts

    def fractions(self) -> torch.Tensor:
        mask = self.total > 0
        fraction = torch.zeros_like(self.active)
        fraction[mask] = self.active[mask] / self.total[mask]
        return fraction

    def reset(self) -> None:
        self.total.zero_()
        self.active.zero_()


class PolicyAutoBuilder(Policy):
    """Generic policy builder for use with configs."""

    def __init__(self, game_rules: GameRules, config: Config | None = None):
        super().__init__()
        self.config = config

        self.components = OrderedDict()
        for component_config in self.config.components:
            name = component_config.name
            self.components[name] = component_config.make_component(game_rules)

        self.action_probs = self.config.action_probs_config.make_component()
        self.network = TensorDictSequential(self.components, inplace=True)
        self._sdpa_context = ExitStack()

        self._activation_trackers: dict[str, _ActivationTracker] = {}
        self._activation_hooks: list[Any] = []
        self._register_activation_tracking()

        self._total_params = sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad and not isinstance(param, UninitializedParameter)
        )

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        td = self.network(td)
        self.action_probs(td, action)
        # Only flatten values if they exist (GRPO policies don't have critic networks)
        if "values" in td.keys():
            td["values"] = td["values"].flatten()
        return td

    def initialize_to_environment(
        self,
        game_rules: GameRules,
        device: torch.device,
    ):
        self.to(device)
        if device.type == "cuda":
            self._configure_sdp()
            torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
            torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        logs = []
        for _, value in self.components.items():
            if hasattr(value, "initialize_to_environment"):
                logs.append(value.initialize_to_environment(game_rules, device))
        if hasattr(self, "action_probs"):
            if hasattr(self.action_probs, "initialize_to_environment"):
                self.action_probs.initialize_to_environment(game_rules, device)

        for log in logs:
            if log is not None:
                log_on_master(log)

    def _configure_sdp(self) -> None:
        self._sdpa_context.close()
        self._sdpa_context = ExitStack()

        configured = False

        nn_attention = getattr(torch.nn, "attention", None)
        sdpa_kernel = getattr(nn_attention, "sdpa_kernel", None)
        if callable(sdpa_kernel):
            configured = self._enter_sdp_context(
                sdpa_kernel,
                backends=[
                    nn_attention.SDPBackend.FLASH_ATTENTION,
                    nn_attention.SDPBackend.EFFICIENT_ATTENTION,
                    nn_attention.SDPBackend.MATH,
                ],
            )

        if configured:
            return

        cuda_backends = getattr(torch.backends, "cuda", None)
        sdp_kernel = getattr(cuda_backends, "sdp_kernel", None) if cuda_backends else None
        if callable(sdp_kernel):
            configured = self._enter_sdp_context(
                sdp_kernel,
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=True,
            )

        if configured or not cuda_backends:
            return

        for attr in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(cuda_backends, attr, None)
            if callable(fn):
                fn(True)

    def _enter_sdp_context(self, fn, *args, **kwargs) -> bool:
        try:
            self._sdpa_context.enter_context(fn(*args, **kwargs))
            return True
        except RuntimeError:
            return False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_sdpa_context"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._sdpa_context = ExitStack()

    def reset_memory(self):
        for _, value in self.components.items():
            if hasattr(value, "reset_memory"):
                value.reset_memory()

    @property
    def total_params(self):
        if hasattr(self, "_total_params"):
            return self._total_params

        params = list(self.parameters())
        skipped_lazy_params = sum(isinstance(param, UninitializedParameter) for param in params)
        self._total_params = sum(param.numel() for param in params if not isinstance(param, UninitializedParameter))

        if skipped_lazy_params:
            log_on_master(
                "Skipped %d uninitialized parameters when logging model size.",
                skipped_lazy_params,
            )

        return self._total_params

    def get_agent_experience_spec(self) -> Composite:
        spec = Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )
        for layer in self.components.values():
            if hasattr(layer, "get_agent_experience_spec"):
                spec.update(layer.get_agent_experience_spec())

        return spec

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # Activation tracking -------------------------------------------------

    def _register_activation_tracking(self) -> None:
        for handle in self._activation_hooks:
            handle.remove()
        self._activation_hooks.clear()
        self._activation_trackers.clear()

        for name, module in self.network.named_modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self._activation_hooks.append(hook)

    def _make_activation_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return

            tracker = self._activation_trackers.get(name)
            features = tensor.shape[-1] if tensor.ndim >= 1 else 1
            if tracker is None:
                tracker = _ActivationTracker.new(features)
                self._activation_trackers[name] = tracker

            tracker.update(tensor)

        return hook

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = PolicyAutoBuilder._extract_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(output, TensorDictBase):
            for value in output.values():
                tensor = PolicyAutoBuilder._extract_tensor(value)
                if tensor is not None:
                    return tensor
        return None

    def get_activation_rate_metrics(self, *, reset: bool = False) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, tracker in self._activation_trackers.items():
            fractions = tracker.fractions()
            for idx, value in enumerate(fractions.tolist()):
                metrics[f"activations/relu/{name}/{idx}"] = float(value)
            if reset:
                tracker.reset()
        return metrics
