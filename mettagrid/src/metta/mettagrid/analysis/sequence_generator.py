"""
Sequence generation utilities for mettagrid analysis.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class SequenceType(Enum):
    """Types of sequences that can be generated."""

    EXTRACTED = "extracted"  # From policy rollouts
    PROCEDURAL = "procedural"  # Hand-crafted strategies
    RANDOM = "random"  # Random exploration


@dataclass
class Sequence:
    """Represents a sequence of observations and actions."""

    observations: List[Any]
    actions: List[int]
    rewards: List[float]
    sequence_type: SequenceType
    strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SequenceExtractor:
    """
    Extracts sequences from trained policies by running rollouts.

    This class generates sequences by executing policies in environments
    and recording the observation-action sequences.
    """

    def __init__(self, max_sequence_length: int = 50, min_sequence_length: int = 10):
        """
        Initialize the sequence extractor.

        Args:
            max_sequence_length: Maximum length of extracted sequences
            min_sequence_length: Minimum length of extracted sequences
        """
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

    def extract_sequences(
        self, policy: nn.Module, environment, num_sequences: int = 100, strategy_filter: Optional[str] = None
    ) -> List[Sequence]:
        """
        Extract sequences from policy rollouts.

        Args:
            policy: Policy to extract sequences from
            environment: Environment to run rollouts in
            num_sequences: Number of sequences to extract
            strategy_filter: Optional strategy to filter for

        Returns:
            List of extracted sequences
        """
        sequences = []
        policy.eval()

        with torch.no_grad():
            for _i in range(num_sequences):
                sequence = self._extract_single_sequence(policy, environment)
                if sequence is not None:
                    sequences.append(sequence)

        return sequences

    def _extract_single_sequence(self, policy: nn.Module, environment) -> Optional[Sequence]:
        """
        Extract a single sequence from a policy rollout.

        Args:
            policy: Policy to extract from
            environment: Environment to run in

        Returns:
            Extracted sequence or None if extraction failed
        """
        # Reset environment
        obs = environment.reset()

        observations = []
        actions = []
        rewards = []

        # Run rollout
        for step in range(self.max_sequence_length):
            # Convert observation to tensor
            if isinstance(obs, (list, tuple)):
                _obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            else:
                _obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs

            # Get action from policy
            # This is a placeholder - actual implementation depends on
            # how the policy processes observations and outputs actions
            # action_logits = policy(obs_tensor)
            # action = torch.argmax(action_logits, dim=-1).item()

            # Placeholder action for now
            action = random.randint(0, 3)  # Assuming 4 actions

            # Step environment
            # next_obs, reward, done, info = environment.step(action)

            # Placeholder environment step
            next_obs = obs  # Placeholder
            reward = 0.0  # Placeholder
            done = step >= self.min_sequence_length  # Placeholder

            # Record step
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

            if done:
                break

        # Check if sequence meets minimum length
        if len(observations) < self.min_sequence_length:
            return None

        # Determine strategy based on sequence characteristics
        strategy = self._classify_strategy(observations, actions, rewards)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.EXTRACTED,
            strategy=strategy,
            metadata={
                "length": len(observations),
                "total_reward": sum(rewards),
                "avg_reward": np.mean(rewards),
                "action_distribution": self._get_action_distribution(actions),
            },
        )

    def _classify_strategy(self, observations: List[Any], actions: List[int], rewards: List[float]) -> str:
        """Classify the strategy used in this sequence."""
        # Simple strategy classification based on actions and rewards
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        total_reward = sum(rewards)
        _avg_reward = np.mean(rewards)

        # Simple heuristics for strategy classification
        if total_reward > 0:
            return "goal_seeking"
        elif len(set(actions)) > 2:
            return "exploration"
        else:
            return "repetitive"

    def _get_action_distribution(self, actions: List[int]) -> Dict[int, int]:
        """Get distribution of actions in sequence."""
        distribution = {}
        for action in actions:
            distribution[action] = distribution.get(action, 0) + 1
        return distribution


class ProceduralSequenceGenerator:
    """
    Generates procedural sequences based on specific strategies.

    This class creates hand-crafted sequences that correspond to
    identifiable strategies or behaviors.
    """

    def __init__(self):
        """Initialize the procedural sequence generator."""
        self.strategies = {
            "goal_seeking": self._generate_goal_seeking_sequence,
            "exploration": self._generate_exploration_sequence,
            "resource_collection": self._generate_resource_collection_sequence,
            "combat": self._generate_combat_sequence,
            "avoidance": self._generate_avoidance_sequence,
        }

    def generate_sequences(self, strategy: str, num_sequences: int = 10, sequence_length: int = 30) -> List[Sequence]:
        """
        Generate procedural sequences for a specific strategy.

        Args:
            strategy: Strategy to generate sequences for
            num_sequences: Number of sequences to generate
            sequence_length: Length of each sequence

        Returns:
            List of generated sequences
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        sequences = []
        for _i in range(num_sequences):
            sequence = self.strategies[strategy](sequence_length)
            sequences.append(sequence)

        return sequences

    def _generate_goal_seeking_sequence(self, length: int) -> Sequence:
        """Generate a goal-seeking sequence."""
        # Simulate goal-seeking behavior: move toward target
        observations = []
        actions = []
        rewards = []

        for i in range(length):
            # Simulate moving toward goal
            obs = self._create_goal_seeking_observation(i, length)
            action = 0  # Move forward
            reward = 1.0 if i > length * 0.8 else 0.1  # Higher reward near end

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.PROCEDURAL,
            strategy="goal_seeking",
            metadata={"length": length, "strategy": "goal_seeking"},
        )

    def _generate_exploration_sequence(self, length: int) -> Sequence:
        """Generate an exploration sequence."""
        observations = []
        actions = []
        rewards = []

        for i in range(length):
            # Simulate exploration behavior: varied actions
            obs = self._create_exploration_observation(i)
            action = random.randint(0, 3)  # Random exploration
            reward = 0.1  # Small reward for exploration

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.PROCEDURAL,
            strategy="exploration",
            metadata={"length": length, "strategy": "exploration"},
        )

    def _generate_resource_collection_sequence(self, length: int) -> Sequence:
        """Generate a resource collection sequence."""
        observations = []
        actions = []
        rewards = []

        for i in range(length):
            # Simulate resource collection: move to resources
            obs = self._create_resource_observation(i)
            action = 1 if i % 3 == 0 else 0  # Collect every 3rd step
            reward = 2.0 if i % 3 == 0 else 0.1  # High reward for collection

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.PROCEDURAL,
            strategy="resource_collection",
            metadata={"length": length, "strategy": "resource_collection"},
        )

    def _generate_combat_sequence(self, length: int) -> Sequence:
        """Generate a combat sequence."""
        observations = []
        actions = []
        rewards = []

        for i in range(length):
            # Simulate combat behavior: aggressive actions
            obs = self._create_combat_observation(i)
            action = 2  # Attack action
            reward = 3.0 if i % 2 == 0 else 0.5  # High reward for attacks

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.PROCEDURAL,
            strategy="combat",
            metadata={"length": length, "strategy": "combat"},
        )

    def _generate_avoidance_sequence(self, length: int) -> Sequence:
        """Generate an avoidance sequence."""
        observations = []
        actions = []
        rewards = []

        for i in range(length):
            # Simulate avoidance behavior: move away from threats
            obs = self._create_avoidance_observation(i)
            action = 3  # Retreat action
            reward = 1.0 if i % 2 == 0 else 0.1  # Reward for successful avoidance

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

        return Sequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            sequence_type=SequenceType.PROCEDURAL,
            strategy="avoidance",
            metadata={"length": length, "strategy": "avoidance"},
        )

    def _create_goal_seeking_observation(self, step: int, total_length: int) -> np.ndarray:
        """Create observation for goal-seeking behavior."""
        # Simulate moving toward goal
        progress = step / total_length
        return np.array([progress, 0.0, 0.0])  # Progress toward goal

    def _create_exploration_observation(self, step: int) -> np.ndarray:
        """Create observation for exploration behavior."""
        # Simulate varied exploration
        return np.array([random.random(), random.random(), 0.0])

    def _create_resource_observation(self, step: int) -> np.ndarray:
        """Create observation for resource collection."""
        # Simulate resource proximity
        resource_proximity = 1.0 if step % 3 == 0 else 0.2
        return np.array([0.0, resource_proximity, 0.0])

    def _create_combat_observation(self, step: int) -> np.ndarray:
        """Create observation for combat behavior."""
        # Simulate enemy proximity
        enemy_proximity = 0.8 if step % 2 == 0 else 0.3
        return np.array([0.0, 0.0, enemy_proximity])

    def _create_avoidance_observation(self, step: int) -> np.ndarray:
        """Create observation for avoidance behavior."""
        # Simulate threat proximity
        threat_proximity = 0.9 if step % 2 == 0 else 0.1
        return np.array([0.0, 0.0, threat_proximity])
