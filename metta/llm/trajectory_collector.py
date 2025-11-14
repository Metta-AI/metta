"""Collect diverse trajectories with rewards for Decision Transformer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mettagrid import MettaGridConfig, Simulator
from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.simulator import Action, AgentObservation

if TYPE_CHECKING:
    from metta.llm.observation_encoder import ObservationEncoder


@dataclass
class Step:
    """Single step in a trajectory."""

    observation: AgentObservation
    observation_text: str  # Encoded for LLM
    action: Action
    action_name: str
    reward: float
    done: bool


@dataclass
class Episode:
    """Complete episode trajectory with returns."""

    steps: list[Step]
    total_reward: float
    returns_to_go: list[float]  # Return-to-go at each step
    policy_name: str
    episode_id: int


class TrajectoryCollector:
    """Collect diverse trajectories with rewards for Decision Transformer training."""

    def __init__(
        self,
        config: MettaGridConfig,
        observation_encoder: "ObservationEncoder",
    ):
        self.config = config
        self.encoder = observation_encoder
        self.simulator = Simulator()

    def collect_episodes(
        self,
        policy: MultiAgentPolicy,
        num_episodes: int,
        policy_name: str = "unknown",
    ) -> list[Episode]:
        """Collect multiple episodes from a policy."""
        episodes = []

        for ep_id in range(num_episodes):
            episode = self.collect_episode(policy, policy_name, ep_id)
            episodes.append(episode)

            if (ep_id + 1) % 10 == 0:
                avg_return = np.mean([e.total_reward for e in episodes[-10:]])
                print(
                    f"Collected {ep_id + 1}/{num_episodes} episodes, " f"avg return (last 10): {avg_return:.2f}"
                )

        return episodes

    def collect_episode(
        self,
        policy: MultiAgentPolicy,
        policy_name: str,
        episode_id: int,
    ) -> Episode:
        """Collect a single episode with full trajectory."""
        # Initialize simulation
        sim = self.simulator.new_simulation(self.config, seed=episode_id)
        agents = sim.agents()
        agent_policies = [policy.agent_policy(i) for i in range(len(agents))]

        # Reset policies
        for p in agent_policies:
            p.reset()

        steps = []
        rewards = []

        # Run episode
        while not sim.is_done():
            for i, agent in enumerate(agents):
                obs = agent.observation

                # Get action from policy
                action = agent_policies[i].step(obs)

                # Set action
                agent.set_action(action)

            # Step simulation
            sim.step()

            # Record transitions (using first agent for now - can extend to multi-agent)
            agent_idx = 0
            obs = agents[agent_idx].observation
            action = agents[agent_idx].last_action  # Get the action we just took
            reward = agents[agent_idx].last_reward  # Reward from that action

            step = Step(
                observation=obs,
                observation_text=self.encoder.encode(obs),
                action=action,
                action_name=action.name,
                reward=reward,
                done=sim.is_done(),
            )
            steps.append(step)
            rewards.append(reward)

        # Compute returns-to-go (cumulative future rewards from each step)
        returns_to_go = []
        running_return = 0.0
        for r in reversed(rewards):
            running_return += r
            returns_to_go.insert(0, running_return)

        return Episode(
            steps=steps,
            total_reward=sum(rewards),
            returns_to_go=returns_to_go,
            policy_name=policy_name,
            episode_id=episode_id,
        )

    def collect_diverse_dataset(
        self,
        policy_uris: dict[str, str],  # {name: uri}
        episodes_per_policy: int,
    ) -> list[Episode]:
        """Collect trajectories from multiple policies to get diverse returns."""
        from metta.rl.checkpoint_manager import CheckpointManager

        all_episodes = []

        for policy_name, policy_uri in policy_uris.items():
            print(f"\nCollecting from policy: {policy_name}")

            # Load policy
            checkpoint_mgr = CheckpointManager(policy_uri)
            policy = checkpoint_mgr.load_policy(self.config)

            # Collect episodes
            episodes = self.collect_episodes(
                policy=policy,
                num_episodes=episodes_per_policy,
                policy_name=policy_name,
            )
            all_episodes.extend(episodes)

            print(
                f"  Collected {len(episodes)} episodes, "
                f"return range: [{min(e.total_reward for e in episodes):.1f}, "
                f"{max(e.total_reward for e in episodes):.1f}]"
            )

        return all_episodes
