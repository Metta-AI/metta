#!/usr/bin/env -S uv run
# Runs policies with ASCII rendering to visualize agent behavior in real-time.


import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.curriculum import SingleTaskCurriculum
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.hydra import get_cfg


class RandomPolicy:
    """Simple random policy for demonstration."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def predict(self, obs):
        """Return random actions for all agents."""
        action = self.action_space.sample()
        return action


class SimplePolicy:
    """A simple policy that tries to move towards objectives."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.cardinal_directions = [1, 3, 5, 7]  # up, left, right, down
        self.move_directions = [1, 2, 3, 5, 7, 8]  # Cardinal + diagonal

    def predict(self, obs):
        """Return simple movement actions."""
        if np.random.random() < 0.6:
            direction = np.random.choice(self.cardinal_directions)
        elif np.random.random() < 0.8:
            direction = np.random.choice(self.move_directions)
        else:
            direction = 4  # Stay in place

        action_type = np.random.choice([0, 1, 2])
        return [direction, action_type]


def get_policy(policy_type: str, env, cfg: DictConfig):
    """Get a policy based on the specified type."""
    if policy_type == "random":
        return RandomPolicy(env)
    elif policy_type == "simple":
        return SimplePolicy(env)
    elif policy_type == "trained":
        try:
            # Try to load trained policy if available
            from metta.agent.policy_store import PolicyStore

            policy_store = PolicyStore(cfg, None)
            policy_pr = policy_store.policy(cfg.policy_uri)

            class TrainedPolicyWrapper:
                def __init__(self, policy):
                    self.policy = policy

                def predict(self, obs):
                    import torch

                    with torch.no_grad():
                        if isinstance(obs, np.ndarray):
                            obs_tensor = torch.from_numpy(obs).float()
                        else:
                            obs_tensor = obs
                        actions, _ = self.policy.forward(obs_tensor)
                        if isinstance(actions, torch.Tensor):
                            actions = actions.numpy()
                        return actions.tolist()

            return TrainedPolicyWrapper(policy_pr.policy())
        except Exception as e:
            print(f"Failed to load trained policy: {e}")
            print("Falling back to simple policy")
            return SimplePolicy(env)
    else:
        print(f"Unknown policy type '{policy_type}', using simple policy")
        return SimplePolicy(env)


def run_renderer(cfg: DictConfig):
    """Run policy visualization with ASCII rendering."""

    # Create environment with ASCII rendering enabled
    env_cfg = get_cfg("benchmark")
    env_cfg.game.num_agents = cfg.renderer_job.num_agents
    env_cfg.game.max_steps = cfg.renderer_job.max_steps

    if cfg.renderer_job.environment:
        env_cfg.game.map_builder = OmegaConf.create(cfg.renderer_job.environment)

    curriculum = SingleTaskCurriculum("renderer", env_cfg)
    env = MettaGridEnv(curriculum, render_mode="human")

    # Get policy
    policy = get_policy(cfg.renderer_job.policy_type, env, cfg)

    # Reset environment
    obs, info = env.reset()

    total_reward = 0
    step_count = 0

    try:
        for _step in range(cfg.renderer_job.num_steps):
            # Get action and step environment
            actions = policy.predict([obs] if not isinstance(obs, list) else obs)

            try:
                obs, rewards, dones, truncs, info = env.step(actions)
            except AssertionError as e:
                if "Task is already complete" in str(e):
                    # Handle the case where task completion is called multiple times
                    # Reset environment to get a fresh task
                    print("🔄 Episode completed, resetting environment...")
                    obs, info = env.reset()
                    continue
                else:
                    raise  # Re-raise if it's a different assertion error

            # Track rewards
            step_reward = np.sum(rewards) if hasattr(rewards, "__len__") else rewards
            total_reward += step_reward
            step_count += 1

            # Render with ASCII renderer
            env.render()

            # Reset if episode done
            if (hasattr(dones, "any") and dones.any()) or (hasattr(dones, "__len__") and any(dones)) or dones:
                obs, info = env.reset()

            # Optional sleep for visualization
            if cfg.renderer_job.sleep_time > 0:
                import time

                time.sleep(cfg.renderer_job.sleep_time)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        env.close()

    print(f"\n🎯 Final Results: {total_reward:.3f} reward over {step_count:,} steps")


@hydra.main(version_base=None, config_path="../configs", config_name="renderer_job")
def main(cfg: DictConfig):
    run_renderer(cfg)


if __name__ == "__main__":
    main()
