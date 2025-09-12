import logging

import torch
import torch.nn as nn
from tensordict import TensorDict
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
        self.layers = nn.ModuleDict()
        for attr, value in self.config.model_config.items():
            if attr == "obs_shaper":
                self.layers[attr] = value.instantiate(obs_meta)
            else:
                self.layers[attr] = value.instantiate()

        self.network = nn.TensorDictSequential(self.layers)

        self.wants_td = True
        self.action_space = env.single_action_space
        self.out_width = obs_meta["obs_width"]
        self.out_height = obs_meta["obs_height"]

    def forward(self, td: TensorDict):
        return self.network(td)

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

        for log in logs:
            if log is not None:
                log_on_master(log)

    def reset_memory(self):
        for _, value in self.layers.items():
            if hasattr(value, "reset_memory"):
                value.reset_memory()

    @property
    def total_params(self):
        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self._total_params

    def get_agent_experience_spec(self) -> Composite:
        # av change this to cycle through layers and get the spec from each layer
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            last_actions=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def attach_replay_buffer(self, experience: Experience):
        """Losses expect to find a replay buffer in the policy."""
        self.replay = experience


# av need Distributed Metta Agent class to wrap this
