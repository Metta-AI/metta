import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl.data import Composite, UnboundedDiscrete

from metta.common.config.config import Config
from metta.rl.experience import Experience

logger = logging.getLogger("metta_agent")


def log_on_master(*args, **argv):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


class Policy(nn.Module):
    """Generic policy builder for use with configs.

    Requirements of your configs:
        - Must have an 'obs_shaper' attribute and it should have an instantiate method that takes an obs_meta dict.
        - Must have an 'instantiate' method that uses itself as the config to instantiate the layer.
        - Must be in the order the network is to be executed.
    """

    def __init__(self, env, obs_meta: dict, config: Config = None):
        super().__init__()
        self.config = config
        self.wants_td = True
        self.action_space = env.single_action_space
        self.out_width = obs_meta["obs_width"]
        self.out_height = obs_meta["obs_height"]

        self.layers = OrderedDict()
        for attr, value in self.config:
            if not isinstance(value, Config):
                continue

            if attr == "obs_shaper_config":
                self.layers["obs_shaper"] = value.instantiate(obs_meta)
            else:
                if "_config" in attr:
                    attr = attr.replace("_config", "")  # not critical but makes it easier to read
                self.layers[attr] = value.instantiate()

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
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device: torch.device,
        is_training: bool = None,
    ):
        logs = []
        for _, value in self.layers.items():
            if hasattr(value, "initialize_to_environment"):
                logs.append(
                    value.initialize_to_environment(features, action_names, action_max_params, device, is_training)
                )

        self.action_probs.initialize_to_environment(features, action_names, action_max_params, device, is_training)

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
        # av change this to cycle through layers and get the spec from each layer
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            # training_env_ids=UnboundedDiscrete(shape=torch.Size([1]), dtype=torch.long),
        )

    def attach_replay_buffer(self, experience: Experience):
        """Losses expect to find a replay buffer in the policy."""
        self.replay = experience


# av need Distributed Metta Agent class to wrap this
