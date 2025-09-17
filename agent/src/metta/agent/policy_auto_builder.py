import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl.data import Composite, UnboundedDiscrete

from metta.common.config.config import Config

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
        self.wants_td = True

        self.layers = OrderedDict()
        for name, layer_config in self.config:
            if not isinstance(layer_config, Config):
                continue

                # all layers have name attr
                # make in and out keys not defaulted
                # all layers take env as an arg
            if name == "obs_shaper_config":
                self.layers["obs_shaper"] = layer_config.instantiate(env)
            else:
                if "_config" in name:
                    name = name.replace("_config", "")  # not critical but makes it easier to read
                self.layers[name] = layer_config.instantiate()

        assert "obs_shaper" in self.layers, "obs_shaper_config must be in the policy's config"

        # finish with a special layer for action probs becaues its forward needs old actions to be passed in
        # we could replace this by adding old_actions to the td
        self.action_probs = self.config.action_probs_config.instantiate()

        self.network = TensorDictSequential(self.layers, inplace=True)

        # av fix this
        # # A dummy forward pass to initialize any lazy modules (e.g. nn.LazyLinear).
        # with torch.no_grad():
        #     dummy_batch_size = [2, 200]
        #     dummy_td = TensorDict(
        #         {
        #             "env_obs": torch.zeros(*dummy_batch_size, 3, dtype=torch.uint8),
        #         },
        #         batch_size=dummy_batch_size,
        #     )
        #     self.network(dummy_td)

    def forward(self, td: TensorDict, action: torch.Tensor = None):
        self.network(td)
        self.action_probs(td, action)
        td["values"] = td[
            "values"
        ].flatten()  # could update Experience to not need this line but  need to update ppo.py
        return td

    def initialize_to_environment(
        self,
        env,
        device: torch.device,
    ):
        self.to(device)
        logs = []
        for _, value in self.layers.items():
            if hasattr(value, "initialize_to_environment"):
                logs.append(value.initialize_to_environment(env, device))
        if hasattr(self, "action_probs"):
            if hasattr(self.action_probs, "initialize_to_environment"):
                self.action_probs.initialize_to_environment(env, device)

        for log in logs:
            if log is not None:
                log_on_master(log)

    def reset_memory(self):
        for _, value in self.layers.items():
            if hasattr(value, "reset_memory"):
                value.reset_memory()

    @property
    def total_params(self):
        # self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._total_params = 42  # av fix this
        # lazy linear params can't be counted. can't make a dummy fwd pass due to need for initalization to env call.
        return self._total_params

    def get_agent_experience_spec(self) -> Composite:
        spec = Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )

        for layer in self.layers.values():
            if hasattr(layer, "get_agent_experience_spec"):
                spec.update(layer.get_agent_experience_spec())

        return spec
