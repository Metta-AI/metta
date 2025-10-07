import logging
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torch.nn.parameter import UninitializedParameter
from torchrl.data import Composite, UnboundedDiscrete

from mettagrid.base_config import Config

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class PolicyAutoBuilder(nn.Module):
    """Generic policy builder for use with configs.

    Requirements of your configs:
        - Must have an 'obs_shaper' attribute and it should have an instantiate method that takes an env.
        - Must have an 'instantiate' method that uses itself as the config to instantiate the layer.
        - Must be in the order the network is to be executed.
    """

    def __init__(self, env, config: Config = None):
        super().__init__()
        self.config = config

        self.components = OrderedDict()
        for component_config in self.config.components:
            name = component_config.name
            self.components[name] = component_config.make_component(env)

        # finish with a special layer for action probs becaues its forward needs old actions to be passed in
        # we could replace this by adding old_actions to the td
        self.action_probs = self.config.action_probs_config.make_component()

        self.network = TensorDictSequential(self.components, inplace=True)
        self._sdpa_context = ExitStack()

        # PyTorch's nn.Module no longer exposes count_params(); defer to manual
        # aggregation to avoid AttributeError during policy construction.
        self._total_params = sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad and not isinstance(param, UninitializedParameter)
        )

    def forward(self, td: TensorDict, action: torch.Tensor = None):
        self.network(td)

        needs_flatten = td.batch_dims > 1
        td_flat = td.reshape(td.batch_size.numel()) if needs_flatten else td

        action_flat = action
        if action is not None and needs_flatten and action.dim() >= 2:
            leading = td.batch_size.numel()
            trailing_shape = action.shape[action.dim() - 1 :]
            action_flat = action.reshape(leading, *trailing_shape)

        self.action_probs(td_flat, action_flat)
        td_flat["values"] = td_flat["values"].flatten()

        return td_flat

    def initialize_to_environment(
        self,
        env,
        device: torch.device,
    ):
        self.to(device)
        if device.type == "cuda":
            self._configure_sdp()
            torch.set_float32_matmul_precision("medium")
        logs = []
        for _, value in self.components.items():
            if hasattr(value, "initialize_to_environment"):
                logs.append(value.initialize_to_environment(env, device))
        if hasattr(self, "action_probs"):
            if hasattr(self.action_probs, "initialize_to_environment"):
                self.action_probs.initialize_to_environment(env, device)

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
        # ExitStack captures contextmanager generators that cannot be pickled.
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
