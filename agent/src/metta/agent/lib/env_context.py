import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class EnvContextLayer(LayerBase):
    """
    Environment context layer that injects learnable context into the LSTM based on environment type.

    This layer:
    1. Takes the task name from the environment and converts it to a one-hot encoding
    2. Applies a learnable context matrix C to get context vectors: C * E
    3. Scales by a fixed amplitude A (currently 1.0)
    4. Injects this as pre-synaptic current into the LSTM hidden state

    The context matrix C can be analyzed after training to understand environment similarities.
    """

    # Hardcoded environment types from the navigation curriculum
    NAVIGATION_ENV_TYPES = [
        "terrain_maps_nohearts",
        "varied_terrain/balanced_large",
        "varied_terrain/balanced_medium",
        "varied_terrain/balanced_small",
        "varied_terrain/sparse_large",
        "varied_terrain/sparse_medium",
        "varied_terrain/sparse_small",
        "varied_terrain/dense_large",
        "varied_terrain/dense_medium",
        "varied_terrain/dense_small",
        "varied_terrain/maze_large",
        "varied_terrain/maze_medium",
        "varied_terrain/maze_small",
        "varied_terrain/cylinder-world_large",
        "varied_terrain/cylinder-world_medium",
        "varied_terrain/cylinder-world_small",
    ]

    def __init__(self, hidden_size: int, amplitude: float = 1.0, enabled: bool = True, **cfg):
        super().__init__(**cfg)
        self.hidden_size = hidden_size
        self.amplitude = amplitude
        self.enabled = enabled
        self.num_env_types = len(self.NAVIGATION_ENV_TYPES)

        # Create environment type to index mapping
        self.env_type_to_idx = {env_type: idx for idx, env_type in enumerate(self.NAVIGATION_ENV_TYPES)}

        # Environment reference will be set during initialization
        self._env = None

    def set_environment(self, env):
        """Set the environment reference to access task information."""
        self._env = env

    def _make_net(self):
        """Create the learnable context matrix C."""
        # Context matrix: each row is a context vector for an environment type
        # Shape: [num_env_types, hidden_size]
        self.context_matrix = nn.Parameter(torch.randn(self.num_env_types, self.hidden_size) * 0.1)

        # Output shape is the same as input (hidden_size)
        self._out_tensor_shape = [self.hidden_size]

        return None  # No network needed, just the parameter

    def _extract_env_type_from_task_name(self, task_name: str) -> str:
        """Extract environment type from task name like 'terrain=varied_terrain/balanced_large;altar=30'."""
        if task_name.startswith("terrain="):
            # Extract the terrain part before the semicolon
            terrain_part = task_name.split(";")[0]
            env_type = terrain_part.replace("terrain=", "")
            return env_type
        else:
            # Fallback for other task naming schemes
            return task_name

    def _get_env_one_hot(self, task_name: str) -> torch.Tensor:
        """Convert task name to one-hot encoding of environment type."""
        env_type = self._extract_env_type_from_task_name(task_name)

        if env_type in self.env_type_to_idx:
            idx = self.env_type_to_idx[env_type]
        else:
            # Unknown environment type - use first index as fallback
            idx = 0

        # Create one-hot vector
        one_hot = torch.zeros(self.num_env_types, device=self.context_matrix.device)
        one_hot[idx] = 1.0

        return one_hot

    def _forward(self, td: TensorDict) -> TensorDict:
        """Forward pass: inject environment context into the hidden state."""
        # Get the hidden state from the previous layer
        if self._sources is None or len(self._sources) == 0:
            raise ValueError("EnvContextLayer requires a source layer")
        hidden = td[self._sources[0]["name"]]

        if not self.enabled:
            # If disabled, just pass through the hidden state unchanged
            td[self._name] = hidden
            return td

        # Get task name from the environment
        task_name = "terrain_maps_nohearts"  # Default fallback
        if self._env is not None and hasattr(self._env, "_task") and self._env._task is not None:
            task_name = self._env._task.name()

        # Get environment one-hot encoding
        env_one_hot = self._get_env_one_hot(task_name)

        # Apply context matrix: C * E
        # env_one_hot: [num_env_types]
        # context_matrix: [num_env_types, hidden_size]
        # context: [hidden_size]
        context = torch.matmul(env_one_hot, self.context_matrix)

        # Scale by amplitude and add to hidden state
        # hidden: [batch_size, hidden_size] or [batch_size * time_steps, hidden_size]
        # context: [hidden_size] -> broadcast to match hidden shape
        context_scaled = self.amplitude * context

        # Add context to hidden state
        td[self._name] = hidden + context_scaled

        return td

    def get_context_matrix(self) -> torch.Tensor:
        """Get the learned context matrix for analysis."""
        return self.context_matrix.detach()

    def get_env_type_similarities(self) -> torch.Tensor:
        """Get cosine similarities between environment type context vectors."""
        context_matrix = self.get_context_matrix()
        # Normalize context vectors
        context_norm = torch.nn.functional.normalize(context_matrix, p=2, dim=1)
        # Compute cosine similarities
        similarities = torch.matmul(context_norm, context_norm.T)
        return similarities
