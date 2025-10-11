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

from metta.agent.meta_cog.mc import MetaCogAction
from mettagrid.base_config import Config

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class MCPolicyAutoBuilder(nn.Module):
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
        self.mc_action_probs = self.config.mc_action_probs_config.make_component()

        self.network = TensorDictSequential(self.components, inplace=True)
        self._sdpa_context = ExitStack()

        # PyTorch's nn.Module no longer exposes count_params(); defer to manual
        # aggregation to avoid AttributeError during policy construction.
        self._total_params = sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad and not isinstance(param, UninitializedParameter)
        )

        # init top noop mc action
        self.noop_mc_action_1 = MetaCogAction("noop_1")
        self.noop_mc_action_2 = MetaCogAction("noop_2")

        # collect all mc actions from components
        self.mc_actions = [self.noop_mc_action_1, self.noop_mc_action_2]
        self.mc_action_names = ["noop_1", "noop_2"]
        for _, component in self.components.items():
            for attr in vars(component).values():
                if isinstance(attr, MetaCogAction):
                    self.mc_actions.append(attr)
                    self.mc_action_names.append(attr.name)

        # initialize mc action indexes based on num of mc actions
        for i, mc_action in enumerate(self.mc_actions):
            mc_action.initialize(i)

        for _, component in self.components.items():
            if hasattr(component, "mc_initialize_to_environment"):
                component.mc_initialize_to_environment(self.mc_action_names)

    def forward(self, td: TensorDict, action: torch.Tensor = None, mc_actions: torch.Tensor = None):
        mc_td = td.clone()
        self.network(mc_td)
        self.mc_action_probs(mc_td, mc_actions)
        td["mc_act_log_prob"] = mc_td["mc_act_log_prob"]
        td["mc_full_log_probs"] = mc_td["mc_full_log_probs"]

        if action is None: # in rollout or eval
            td["mc_actions"] = mc_td["mc_actions"]
        else: # in training
            td["mc_entropy"] = mc_td["mc_entropy"]

        self.apply_mc_actions(mc_td, td["mc_actions"])

        self.network(td)
        self.action_probs(td, action)

        td["values"] = td["values"].flatten()  # could update Experience to not need this line but need to update ppo.py
        return td

    def apply_mc_actions(self, td: TensorDict, mc_actions: torch.Tensor):
        env_ids = td.get("env_ids", None)
        if env_ids is None:
            env_ids = torch.arange(td.batch_size.numel(), device=mc_actions.device)

        for mc_action in self.mc_actions:
            # Filter env_ids to only those where mc_actions matches this action's index
            mask = mc_actions == mc_action.action_index
            filtered_env_ids = env_ids[mask]

            # Only call if there are matching env_ids
            if filtered_env_ids.numel() > 0:
                mc_action(filtered_env_ids)

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
