import collections
import contextlib
import logging
import typing

import tensordict
import tensordict.nn
import torch
import torch.nn.parameter
import torchrl.data

import metta.agent.policy
import mettagrid.base_config
import mettagrid.policy.policy_env_interface

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class PolicyAutoBuilder(metta.agent.policy.Policy):
    """Generic policy builder for use with configs."""

    def __init__(
        self,
        policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface,
        config: mettagrid.base_config.Config | None = None,
    ):
        super().__init__(policy_env_info)
        self.config = config

        self.components = collections.OrderedDict()
        for component_config in self.config.components:
            name = component_config.name
            self.components[name] = component_config.make_component(policy_env_info)

        self.action_probs = self.config.action_probs_config.make_component()
        self._sequential_network = tensordict.nn.TensorDictSequential(self.components, inplace=True)
        self._sdpa_context = contextlib.ExitStack()

        self._total_params = sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad and not isinstance(param, torch.nn.parameter.UninitializedParameter)
        )

    def forward(self, td: tensordict.TensorDict, action: typing.Optional[torch.Tensor] = None) -> tensordict.TensorDict:
        td = self._sequential_network(td)
        self.action_probs(td, action)
        # Only flatten values if they exist (GRPO policies don't have critic networks)
        if "values" in td.keys():
            td["values"] = td["values"].flatten()
        return td

    def initialize_to_environment(
        self,
        policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface,
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
                logs.append(value.initialize_to_environment(policy_env_info, device))
        if hasattr(self, "action_probs"):
            if hasattr(self.action_probs, "initialize_to_environment"):
                self.action_probs.initialize_to_environment(policy_env_info, device)

        for log in logs:
            if log is not None:
                log_on_master(log)

    def _configure_sdp(self) -> None:
        self._sdpa_context.close()
        self._sdpa_context = contextlib.ExitStack()

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

    def __getstate__(self) -> dict[str, typing.Any]:
        state = self.__dict__.copy()
        state["_sdpa_context"] = None
        return state

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self._sdpa_context = contextlib.ExitStack()

    def reset_memory(self):
        for _, value in self.components.items():
            if hasattr(value, "reset_memory"):
                value.reset_memory()

    @property
    def total_params(self):
        if hasattr(self, "_total_params"):
            return self._total_params

        params = list(self.parameters())
        skipped_lazy_params = sum(isinstance(param, torch.nn.parameter.UninitializedParameter) for param in params)
        self._total_params = sum(
            param.numel() for param in params if not isinstance(param, torch.nn.parameter.UninitializedParameter)
        )

        if skipped_lazy_params:
            log_on_master(
                "Skipped %d uninitialized parameters when logging model size.",
                skipped_lazy_params,
            )

        return self._total_params

    def get_agent_experience_spec(self) -> torchrl.data.Composite:
        spec = torchrl.data.Composite(
            env_obs=torchrl.data.UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )
        for layer in self.components.values():
            if hasattr(layer, "get_agent_experience_spec"):
                spec.update(layer.get_agent_experience_spec())

        return spec

    def network(self) -> torch.nn.Module:
        """Return the sequential component stack used for inference and training."""
        return self._sequential_network

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
