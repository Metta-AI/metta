import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class EnvironmentalContextEmbedding(LayerBase):
    """
    Environmental context embedding layer that learns embeddings for task IDs.

    This layer takes a task ID hash and learns an embedding that can be used to
    condition the agent's behavior based on the environment it's in. The embedding
    can be added to the observation stream or used to condition the LSTM initial state.

    Args:
        embedding_dim: Dimension of the learned embeddings (should match LSTM hidden size)
        num_task_embeddings: Number of possible task embeddings (upper bound for task hash space)
        strategy: How to integrate the embedding - "input_sum" or "initial_state"
    """

    def __init__(self, embedding_dim: int = 128, num_task_embeddings: int = 1000, strategy: str = "input_sum", **cfg):
        self.embedding_dim = embedding_dim
        self.num_task_embeddings = num_task_embeddings
        self.strategy = strategy

        # Validate strategy
        if strategy not in ["input_sum", "initial_state"]:
            raise ValueError(f"Strategy must be 'input_sum' or 'initial_state', got {strategy}")

        super().__init__(**cfg)

    def _initialize(self):
        """Initialize the task embedding layer."""
        # Create embedding layer for task IDs
        self.task_embeddings = nn.Embedding(num_embeddings=self.num_task_embeddings, embedding_dim=self.embedding_dim)

        # Initialize embeddings with small random values
        nn.init.normal_(self.task_embeddings.weight, mean=0.0, std=0.1)

        # Set output tensor shape
        self._out_tensor_shape = [self.embedding_dim]

    def _forward(self, td: TensorDict):
        """
        Forward pass to generate environmental context embedding.

        Expects task_id in the tensor dict from the environment.
        For strategy "input_sum", adds the embedding to the encoded observation.
        For strategy "initial_state", stores the embedding for LSTM initialization.
        """
        # Get task ID from tensor dict (should be provided by environment)
        if "task_id" not in td:
            # If no task ID provided, use zero embedding (no environmental context)
            task_id = torch.zeros(td["x"].shape[0], dtype=torch.long, device=td["x"].device)
        else:
            task_id = td["task_id"]

        # Generate embedding for the task ID
        context_embedding = self.task_embeddings(task_id)

        if self.strategy == "input_sum":
            # Add the context embedding to the encoded observation
            # The input should be from the encoded_obs layer (shape: [batch_size, 128])
            # We need to find the correct input tensor from our sources
            input_tensor = None

            # Look for the input from our source components
            if self._source_components is not None:
                for source_name, _source_component in self._source_components.items():
                    if source_name in td:
                        input_tensor = td[source_name]
                        break

            # If we can't find the source tensor, it means the source hasn't been processed yet
            # This should not happen now that we process environmental context after observation processing
            if input_tensor is None:
                raise ValueError(f"Source tensor not found in TensorDict. Available keys: {list(td.keys())}")

            # Add the context embedding to the encoded observation
            td[self._name] = input_tensor + context_embedding
        elif self.strategy == "initial_state":
            # Store the context embedding for LSTM initialization
            # This will be used by the LSTM layer to condition its initial state
            td[self._name] = context_embedding
            # Also store in a special key for LSTM to access
            td["_environmental_context"] = context_embedding

        return td

    def get_task_embedding(self, task_id: torch.Tensor) -> torch.Tensor:
        """Get embedding for a specific task ID (useful for analysis)."""
        return self.task_embeddings(task_id)

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all learned task embeddings (useful for analysis)."""
        return self.task_embeddings.weight.data
