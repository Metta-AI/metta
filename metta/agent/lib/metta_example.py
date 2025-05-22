"""
Example usage of the Metta Neural Network Architecture.

This example shows how to build a simple LSTM-based policy network
using the refactored MettaModule and MettaSystem classes with proper state management.
"""

from typing import Dict, List

import torch
from metta_architecture_refactored import LinearModule, LSTMModule, MettaModule, MettaSystem


class EmbeddingModule(MettaModule):
    """Module for embedding discrete inputs."""

    def __init__(self, name: str, embedding_dim: int, num_embeddings: int, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def _calculate_output_shape(self):
        """Calculate output shape based on input shapes."""
        self._out_tensor_shape = [self.embedding_dim]

    def _make_net(self):
        """Create embedding network."""
        return torch.nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, inputs: List[torch.Tensor]):
        """Forward pass with input tensors."""
        if not inputs or self._net is None:
            raise RuntimeError("Invalid inputs or uninitialized module")

        # Expect integer input tensor
        indices = inputs[0].long()
        return self._net(indices)


class PolicyNetwork:
    """A simple policy network using the refactored Metta architecture with proper state management."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        # Create the MettaSystem (includes graph, executor, recreation manager)
        self._system = MettaSystem()

        # Create modules
        self._encoder = LinearModule("encoder", output_size=hidden_dim)
        self._core = LSTMModule("core", hidden_size=hidden_dim)
        self._action_head = LinearModule("action_head", output_size=action_dim)
        self._value_head = LinearModule("value_head", output_size=1)

        # Add modules to system
        self._system.add_module(self._encoder)
        self._system.add_module(self._core)
        self._system.add_module(self._action_head)
        self._system.add_module(self._value_head)

        # Connect modules
        self._system.connect("encoder", "core")
        self._system.connect("core", "action_head")
        self._system.connect("core", "value_head")

        # Setup the system with an initial observation shape
        self._setup_with_dummy_input(obs_dim)

    @property
    def system(self):
        """Get the MettaSystem instance."""
        return self._system

    @property
    def graph(self):
        """Get the MettaGraph instance."""
        return self._system.graph

    @property
    def encoder(self):
        """Get the encoder module."""
        return self._encoder

    @property
    def core(self):
        """Get the core module."""
        return self._core

    @property
    def action_head(self):
        """Get the action head module."""
        return self._action_head

    @property
    def value_head(self):
        """Get the value head module."""
        return self._value_head

    def _setup_with_dummy_input(self, obs_dim: int):
        """Setup the system with dummy input to initialize shapes."""
        # Set up the encoder's input shapes (it has no source modules)
        self._encoder._in_tensor_shapes = [[obs_dim]]

        # Setup the entire system (this provides state management functions to modules)
        self._system.setup()

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the policy network with proper state management."""
        # Create inputs dictionary
        inputs = {"encoder": obs}

        # Run forward pass through the system (includes proper state management)
        outputs = self._system.forward(inputs)

        return {"action_logits": outputs["action_head"], "value": outputs["value_head"]}

    def reset_states(self):
        """Reset LSTM states (useful for episode boundaries)."""
        self._system.reset_states()

    def get_lstm_state(self):
        """Get current LSTM state for inspection."""
        return self._system.executor.get_state("core")


def demonstrate_state_management():
    """Demonstrate that LSTM state is properly managed."""
    print("=== LSTM State Management Demonstration ===")

    # Create policy network
    policy = PolicyNetwork(obs_dim=24, action_dim=4, hidden_dim=64)

    # Create sequence of observations
    batch_size = 1
    sequence_length = 3
    obs_dim = 24

    print(f"Initial LSTM state: {policy.get_lstm_state()}")

    for step in range(sequence_length):
        # Create observation for this step
        obs = torch.randn(batch_size, obs_dim)

        # Forward pass
        outputs = policy.forward(obs)

        # Check LSTM state
        lstm_state = policy.get_lstm_state()
        print(f"Step {step + 1} - LSTM state exists: {lstm_state is not None}")
        if lstm_state is not None:
            print(f"  Hidden state shape: {lstm_state[0].shape}")
            print(f"  Cell state shape: {lstm_state[1].shape}")

    # Reset states
    print("\nResetting states...")
    policy.reset_states()
    print(f"After reset - LSTM state: {policy.get_lstm_state()}")


def demonstrate_dynamic_capabilities():
    """Demonstrate dynamic recreation capabilities."""
    print("\n=== Dynamic Capabilities Demonstration ===")

    # Create policy network
    policy = PolicyNetwork(obs_dim=24, action_dim=4, hidden_dim=64)

    # Run initial forward pass
    obs = torch.randn(2, 24)
    initial_output = policy.forward(obs)
    print(f"Initial action logits shape: {initial_output['action_logits'].shape}")

    # Dynamically resize action head for more actions
    print("\nResizing action head from 4 to 8 actions...")
    policy.system.resize_module_output("action_head", new_output_size=8)

    # Run forward pass with new action space
    new_output = policy.forward(obs)
    print(f"New action logits shape: {new_output['action_logits'].shape}")

    # Show that weights were preserved where possible
    print("✓ Network successfully adapted to new action space")


def main():
    """Run comprehensive examples of the policy network."""
    # Basic usage example
    print("=== Basic Usage Example ===")

    # Create a policy network
    batch_size = 16
    obs_dim = 24
    action_dim = 4

    policy = PolicyNetwork(obs_dim, action_dim)

    # Create a random observation
    obs = torch.rand(batch_size, obs_dim)

    # Run forward pass
    outputs = policy.forward(obs)

    # Print output shapes
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")

    # Sample actions
    action_probs = torch.softmax(outputs["action_logits"], dim=-1)
    actions = torch.multinomial(action_probs, num_samples=1)

    print(f"Sampled actions shape: {actions.shape}")

    # Demonstrate state management
    demonstrate_state_management()

    # Demonstrate dynamic capabilities
    demonstrate_dynamic_capabilities()


if __name__ == "__main__":
    main()
