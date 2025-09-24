"""Policy interfaces and implementations for CoGames."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from mettagrid.envs.mettagrid_env import MettaGridEnv


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def get_action(self, observation: Any, agent_id: Optional[int] = None) -> Any:
        """Get action for given observation.

        Args:
            observation: The current observation from the environment
            agent_id: Optional agent ID for multi-agent scenarios

        Returns:
            The action to take
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy state."""
        pass

    def save(self, path: str) -> None:
        """Save the policy to disk.

        Args:
            path: Path to save the policy
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support saving")

    @classmethod
    def load(cls, path: str) -> "Policy":
        """Load a policy from disk.

        Args:
            path: Path to load the policy from

        Returns:
            The loaded policy
        """
        raise NotImplementedError(f"{cls.__name__} does not support loading")


class RandomPolicy(Policy):
    """Random policy that samples actions uniformly from the action space."""

    def __init__(self, env: MettaGridEnv):
        """Initialize random policy.

        Args:
            env: The environment to sample actions from
        """
        self.env = env
        self.action_space = env.action_space

    def get_action(self, observation: Any, agent_id: Optional[int] = None) -> Any:
        """Get a random action.

        Args:
            observation: The current observation (ignored for random policy)
            agent_id: Optional agent ID (ignored for random policy)

        Returns:
            A random action sampled from the action space
        """
        return self.action_space.sample()

    def reset(self) -> None:
        """Reset the policy state (no-op for random policy)."""
        pass


class TrainedPolicy(Policy):
    """Policy loaded from a trained checkpoint."""

    def __init__(self, checkpoint_path: str, env: Optional[MettaGridEnv] = None):
        """Initialize policy from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint
            env: Optional environment for policy initialization
        """
        self.checkpoint_path = checkpoint_path
        self.env = env
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from checkpoint."""
        # This would integrate with metta.rl.training
        # For now, it's a placeholder
        raise NotImplementedError("Model loading not yet implemented. Use RandomPolicy for testing.")

    def get_action(self, observation: Any, agent_id: Optional[int] = None) -> Any:
        """Get action from trained model.

        Args:
            observation: The current observation
            agent_id: Optional agent ID for multi-agent scenarios

        Returns:
            The action from the trained model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Placeholder for actual inference
        raise NotImplementedError("Model inference not yet implemented")

    def reset(self) -> None:
        """Reset any internal state of the policy."""
        pass

    def save(self, path: str) -> None:
        """Save the policy checkpoint.

        Args:
            path: Path to save the policy
        """
        # Would save the model checkpoint
        raise NotImplementedError("Model saving not yet implemented")

    @classmethod
    def load(cls, path: str, env: Optional[MettaGridEnv] = None) -> "TrainedPolicy":
        """Load a trained policy from checkpoint.

        Args:
            path: Path to the checkpoint
            env: Optional environment for policy initialization

        Returns:
            The loaded policy
        """
        return cls(path, env)


def create_policy(policy_type: str, env: MettaGridEnv, **kwargs) -> Policy:
    """Factory function to create policies.

    Args:
        policy_type: Type of policy ("random", "trained", or checkpoint path)
        env: The environment for the policy
        **kwargs: Additional arguments for policy creation

    Returns:
        The created policy

    Raises:
        ValueError: If policy type is not recognized
    """
    if policy_type == "random":
        return RandomPolicy(env)
    elif policy_type == "trained" or policy_type.startswith("file://") or policy_type.endswith(".ckpt"):
        checkpoint_path = kwargs.get("checkpoint_path", policy_type)
        return TrainedPolicy(checkpoint_path, env)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
