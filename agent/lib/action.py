import agent.lib.nn_layer_library as nn_layer_library
from tensordict import TensorDict

class ActionType(nn_layer_library.Bilinear):
    def __init__(self, **cfg):
        self._action_type_embeds = [[]] # to be populated with action type embeds at runtime
        self._embed_length = 16 # TODO: make this dynamic
        self._output_size = 1
        super().__init__(**cfg)
        

class ActionParam(nn_layer_library.Bilinear):
    def __init__(self, **cfg):
        self._action_param_embeds = [[]] # to be populated with action param embeds at runtime
        self._embed_length = 16 # TODO: make this dynamic
        self._output_size = 1
        super().__init__(**cfg)

    def _forward(self, td: TensorDict):
        state_features = td[self._input_source]
        """
        Forward pass for scoring each action given state features.
        Args:
            state_features (torch.Tensor): Tensor of shape [N, feature_dim] where
                                           N is the batch size.
        Returns:
            logits (torch.Tensor): Tensor of shape [N, num_actions], representing
                                   the score for each action per state.
        """
        # state_features: [N, feature_dim] (we assume a vector)
        N = state_features.size(0)  # Batch size

        # this could be updated once in trainer.py
        num_actions = len(self._action_param_embeds)  # Total number of actions
        
        # action_embeds: [num_actions, embedding_dim]
        action_embeds = self._action_param_embeds
        
        # Expand state_features to prepare for pairing with each action.
        # Add a dimension so state_features becomes [N, 1, feature_dim],
        # then expand along that dimension to get [N, num_actions, feature_dim].
        state_expanded = state_features.unsqueeze(1).expand(-1, num_actions, -1)
        # state_expanded: [N, num_actions, feature_dim]
        
        # Similarly, expand the action embeddings so that they match the batch dimension.
        # Start with [num_actions, embedding_dim], add a dimension to get [1, num_actions, embedding_dim],
        # then expand to [N, num_actions, embedding_dim].
        action_expanded = action_embeds.unsqueeze(0).expand(N, -1, -1)
        # action_expanded: [N, num_actions, embedding_dim]
        
        # Flatten both tensors so they can be passed to the bilinear layer.
        # Merge the batch and action dimensions:
        # state_flat: [N * num_actions, feature_dim]
        state_flat = state_expanded.reshape(-1, state_features.size(1))
        # action_flat: [N * num_actions, embedding_dim]
        action_flat = action_expanded.reshape(-1, action_embeds.size(1))
        
        # Apply the bilinear layer to each state-action pair.
        # The bilinear layer processes pairs of [feature_dim] and [embedding_dim] to output a scalar.
        # logits_flat: [N * num_actions, 1]
        logits_flat = self.bilinear(state_flat, action_flat)
        
        # Reshape the output logits to [N, num_actions] so each state has a score for each action.
        logits = logits_flat.view(N, num_actions)
        
        return logits
    

        return td

# Example usage:
if __name__ == "__main__":
    # Define dimensions for our example.
    batch_size = 32
    feature_dim = 128
    embedding_dim = 16
    num_actions = 10

    # Create a dummy tensor of state features with shape [batch_size, feature_dim].
    state_features = torch.randn(batch_size, feature_dim)

    # Instantiate the BilinearActionDecoder.
    decoder = BilinearActionDecoder(feature_dim, embedding_dim, num_actions)

    # Forward pass: Compute logits for each state-action pair.
    logits = decoder(state_features)

    # logits will have the shape [batch_size, num_actions].
        
