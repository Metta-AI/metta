import torch
import torch.nn as nn
from einops import repeat
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.rl.training.training_environment import EnvironmentMetaData


class ActionEmbeddingConfig(ComponentConfig):
    out_key: str
    name: str = "action_embedding"
    num_embeddings: int = 100
    embedding_dim: int = 16

    def make_component(self, env=None):
        return ActionEmbedding(config=self)


class ActionEmbedding(nn.Module):
    """
    Creates and manages embeddings for available actions in the environment.

    This class extends the base Embedding layer to specifically handle action embeddings
    in a reinforcement learning context. It maintains a dictionary mapping action names to
    embedding indices, and dynamically updates the set of active actions based on what's
    available in the current environment.

    Key features:
    - Maintains a mapping between action names (strings) and embedding indices
    - Dynamically activates subsets of actions when requested
    - Expands embeddings to match batch dimensions automatically
    - Stores the number of active actions in the TensorDict for other layers

    The initialize_to_environment method should be called whenever the available actions in the
    environment change and after init, providing the new set of action names and the target device.
    """

    def __init__(self, config: ActionEmbeddingConfig):
        super().__init__()
        self.config = config
        self._reserved_action_embeds = {}
        self.num_actions = 0
        self.out_key = self.config.out_key
        self.num_embeddings = self.config.num_embeddings
        self.embedding_dim = self.config.embedding_dim
        self.register_buffer("active_indices", torch.tensor([], dtype=torch.long))
        self.net = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        weight_limit = 0.1
        nn.init.orthogonal_(self.net.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.net.weight))
            self.net.weight.mul_(weight_limit / max_abs_value)

    def initialize_to_environment(
        self,
        env: EnvironmentMetaData,
        device: torch.device,
    ) -> None:
        if not hasattr(env, "action_names") or not hasattr(env, "max_action_args"):
            raise AttributeError(
                "Environment metadata must provide 'action_names' and 'max_action_args' to initialize action embeddings"
            )

        base_action_names = list(env.action_names)
        action_max_params = list(env.max_action_args)

        action_names = [
            f"{name}_{i}"
            for name, max_param in zip(base_action_names, action_max_params, strict=False)
            for i in range(max_param + 1)
        ]

        for action_name in action_names:
            if action_name not in self._reserved_action_embeds:
                embedding_index = len(self._reserved_action_embeds) + 1  # generate index for this string
                self._reserved_action_embeds[action_name] = embedding_index  # update this component's known embeddings

        self.active_indices = torch.tensor(
            [self._reserved_action_embeds[action_name] for action_name in action_names], device=device, dtype=torch.long
        )
        self.num_actions = len(self.active_indices)

    def forward(self, td: TensorDict):
        B_TT = td.batch_size.numel()

        # get embeddings then expand to match the batch size
        td[self.out_key] = repeat(self.net(self.active_indices), "a e -> b a e", b=B_TT)
        return td
