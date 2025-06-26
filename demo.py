#!/usr/bin/env python3
"""
Minimal standalone training demo for Metta

This script shows the core training loop without Hydra configuration.
It directly creates the necessary components for training.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.rl.experience import Experience
from metta.rl.functional_trainer import train_ppo
from metta.rl.losses import Losses


def main():
    print("Metta Minimal Training Demo")
    print("=" * 60)
    print("\nNOTE: This demo shows the training loop structure.")
    print("It will fail when trying to compute advantages because it requires")
    print("pufferlib's C++ operations. This is expected - the goal is to show")
    print("how to structure the training code without Hydra.\n")

    # Basic configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Training parameters
    total_timesteps = 10_000
    batch_size = 2048
    minibatch_size = 256
    bptt_horizon = 16
    learning_rate = 0.0003

    # Create a dummy environment for demonstration
    # In a real scenario, you would create a proper MettaGridEnv here
    print("\nNote: This is a minimal demo showing the training loop structure.")
    print("To use with actual environments, replace the dummy components with real ones.")

    # Dummy components for demonstration
    class DummyEnv:
        """Minimal dummy environment for demonstration"""

        def __init__(self):
            self.num_agents = 12
            self.single_observation_space = type("Space", (), {"shape": (11, 11, 4), "dtype": np.float32})()
            self.single_action_space = type(
                "Space",
                (),
                {
                    "n": 8,
                    "dtype": np.int32,
                    "shape": (2,),  # action type and parameter
                },
            )()
            self.agents_per_batch = None
            self.action_names = ["noop", "move", "rotate", "attack"]
            self.max_action_args = [0, 1, 1, 1]

    class DummyVecEnv:
        """Minimal dummy vectorized environment"""

        def __init__(self, env):
            self.driver_env = env
            self.num_agents = env.num_agents * 4  # 4 environments
            self.single_observation_space = env.single_observation_space
            self.single_action_space = env.single_action_space

        def async_reset(self, seed):
            pass

        def recv(self):
            # Return dummy data
            obs = np.random.randn(self.num_agents, *self.driver_env.single_observation_space.shape)
            rewards = np.zeros(self.num_agents)
            dones = np.zeros(self.num_agents, dtype=bool)
            truncs = np.zeros(self.num_agents, dtype=bool)
            infos = {}
            env_ids = list(range(self.num_agents))
            masks = np.ones(self.num_agents, dtype=bool)
            return obs, rewards, dones, truncs, infos, env_ids, masks

        def send(self, actions):
            pass

        def close(self):
            pass

    # Create dummy environment
    env = DummyEnv()
    vecenv = DummyVecEnv(env)
    vecenv.async_reset(42)

    # Create policy store
    cfg = DictConfig({"device": str(device), "data_dir": "./demo_run"})
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a dummy policy
    class DummyPolicy(torch.nn.Module):
        """Minimal policy for demonstration"""

        def __init__(self, obs_shape, num_actions):
            super().__init__()
            self.obs_shape = obs_shape
            self.num_actions = num_actions
            self.hidden_size = 128

            # Simple architecture
            self.features = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(obs_shape), self.hidden_size),
                torch.nn.ReLU(),
            )
            self.actor = torch.nn.Linear(self.hidden_size, num_actions)
            self.critic = torch.nn.Linear(self.hidden_size, 1)

        def forward(self, obs, state=None, action=None):
            features = self.features(obs)
            logits = self.actor(features)
            value = self.critic(features)

            # Simple action sampling
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if action is None:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action[..., 0])  # Assume first column is action type

            # Dummy second action parameter
            action_full = torch.stack([action, torch.zeros_like(action)], dim=-1).int()

            entropy = dist.entropy()

            return action_full, log_prob, entropy, value, logits

        def activate_actions(self, action_names, max_args, device):
            """Compatibility method"""
            pass

    # Create policy
    policy = DummyPolicy(env.single_observation_space.shape, env.single_action_space.n).to(device)
    policy.activate_actions(env.action_names, env.max_action_args, device)

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=minibatch_size,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=policy.hidden_size,
        cpu_offload=False,
        num_lstm_layers=1,  # At least 1 LSTM layer required
        agents_per_batch=None,
    )

    # Training state
    agent_step = 0
    epoch = 0
    losses = Losses()
    timer = Stopwatch(None)
    timer.start()

    # Dummy config for train_ppo
    agent_cfg = DictConfig({"agent": {"clip_range": 0}})

    print("\nStarting training loop (demo mode)")
    print("=" * 60)

    # Main training loop
    while agent_step < total_timesteps:
        steps_before = agent_step

        # ROLLOUT: Collect experience
        with timer("rollout"):
            # In a real implementation, this would collect actual experience
            # For demo, we just simulate the rollout
            steps_collected = batch_size
            agent_step += steps_collected

            # Simulate filling the experience buffer
            experience.reset_for_rollout()
            while not experience.ready_for_training:
                obs, rewards, dones, truncs, infos, env_ids, masks = vecenv.recv()

                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, log_probs, _, values, _ = policy(obs_tensor)

                experience.store(
                    obs=obs_tensor,
                    actions=actions,
                    logprobs=log_probs,
                    rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                    dones=torch.tensor(dones, dtype=torch.bool, device=device),
                    truncations=torch.tensor(truncs, dtype=torch.bool, device=device),
                    values=values.flatten(),
                    env_id=slice(0, len(env_ids)),
                    mask=torch.tensor(masks, dtype=torch.bool),
                    lstm_state=None,
                )

                vecenv.send(actions.cpu().numpy())

        # TRAIN: Update policy with PPO
        with timer("train"):
            epoch = train_ppo(
                policy=policy,
                optimizer=optimizer,
                experience=experience,
                device=device,
                losses=losses,
                epoch=epoch,
                cfg=agent_cfg,
                lr_scheduler=None,
                timer=timer,
                # PPO hyperparameters
                gamma=0.99,
                gae_lambda=0.95,
                clip_coef=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                norm_adv=True,
                clip_vloss=True,
                vf_clip_coef=0.1,
                update_epochs=4,
                target_kl=None,
                kickstarter=None,
                agent_step=agent_step,
                l2_reg_loss_coef=0.0,
                l2_init_loss_coef=0.0,
                clip_range=0,
                # Additional parameters
                prio_alpha=0.0,
                prio_beta0=0.6,
                total_timesteps=total_timesteps,
                batch_size=batch_size,
                vtrace_rho_clip=1.0,
                vtrace_c_clip=1.0,
            )

        # Log progress
        steps_per_sec = (agent_step - steps_before) / (
            timer.get_last_elapsed("rollout") + timer.get_last_elapsed("train")
        )
        loss_stats = losses.stats()

        print(
            f"Epoch {epoch:4d} | Steps: {agent_step:6d}/{total_timesteps} | "
            f"SPS: {steps_per_sec:5.0f} | Loss: {loss_stats.get('policy_loss', 0):.4f}"
        )

    # Clean up
    vecenv.close()

    print("\n✅ Demo training loop complete!")
    print("To use with real environments:")
    print("1. Replace DummyEnv with MettaGridEnv")
    print("2. Replace DummyPolicy with a real MettaAgent")
    print("3. Configure the environment with proper YAML or dict config")


if __name__ == "__main__":
    main()
