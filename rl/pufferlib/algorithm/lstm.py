import torch
from tensordict import TensorDict

from .algorithm import Algorithm

class LSTM(Algorithm):
    def __init__(self, num_layers, total_agents, hidden_size):
        assert num_layers > 0
        assert total_agents > 0
        assert hidden_size > 0

        self.num_layers = num_layers
        self.total_agents = total_agents
        self.hidden_size = hidden_size

    def make_experience_buffers(self, experience: TensorDict):
        shape = (self.num_layers, self.total_agents, self.hidden_size)
        experience["lstm"] = TensorDict({
            "h": torch.zeros(shape),
            "c": torch.zeros(shape),
        }, device=experience.device, batch_size=shape)

    def on_pre_step(self, experience: TensorDict, state: TensorDict):
        # xcxc can we use ":"
        state["lstm"] = experience["lstm"][:, state["env", "env_id"]]

    def on_post_step(self, experience: TensorDict, state: TensorDict):
        pass

    # xcxc can we use ":"
    def store_experience(self, experience: TensorDict, state: TensorDict):
        experience["lstm"][:, state["env", "env_id"]] = state["next", "lstm"]



