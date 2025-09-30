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

from metta.agent.util.torch_backends import build_sdpa_context
from mettagrid.config import Config

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
        self.action_probs(td, action)
        td["values"] = td["values"].flatten()  # could update Experience to not need this line but need to update ppo.py
        return td

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

        context = build_sdpa_context(
            prefer_flash=True,
            prefer_mem_efficient=True,
            prefer_math=True,
            set_priority=True,
        )

        if context is not None:
            self._sdpa_context.enter_context(context)
        else:
            logger.debug("Unable to configure scaled dot-product attention backends; using PyTorch defaults.")

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
