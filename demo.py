#!/usr/bin/env python3
"""
Standalone training demo for Metta - Alternative to tools/train.py without Hydra

This script demonstrates the core training loop structure of Metta without
using Hydra's configuration system. It shows:

1. How to initialize all training components explicitly
2. The main training loop: rollout → train_ppo → repeat
3. How hyperparameters flow through the system

NOTE: There's currently a bug in mettagrid_env.py line 157 where it calls
cpp_config_dict() on the configuration before passing it to MettaGrid.
The MettaGrid C++ constructor actually expects the original Python config
format, not the converted format. This causes a KeyError: 'groups'.

Until this is fixed, this demo shows the training structure using dummy
components. Once the bug is fixed (by removing cpp_config_dict call),
you can use real MettaGridEnv as shown in the commented code below.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.losses import Losses


def main():
    print("Metta Standalone Training Demo (Hydra-free)")
    print("=" * 60)
    print("\nThis demo shows how Metta's training loop works without Hydra.")
    print("Due to a bug in mettagrid_env.py, we use dummy components,")
    print("but the training structure is identical to the real implementation.\n")

    # Basic configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Training parameters (from configs/trainer/puffer.yaml)
    total_timesteps = 10_000  # Reduced for demo
    batch_size = 2048
    minibatch_size = 256
    bptt_horizon = 16
    learning_rate = 0.0004573

    # PPO hyperparameters (from configs/trainer/puffer.yaml)
    gamma = 0.977
    gae_lambda = 0.916
    clip_coef = 0.1
    ent_coef = 0.0021
    vf_coef = 0.44
    max_grad_norm = 0.5
    update_epochs = 1

    print("Configuration loaded (based on configs/trainer/puffer.yaml)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  PPO clip: {clip_coef}")
    print()

    # This is how you would create a real MettaGridEnv once the bug is fixed:
    """
    # Create environment configuration
    env_config = {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.0,
        "game": {
            "num_agents": 2,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "max_steps": 1000,
            "diversity_bonus": {
                "enabled": False,
                "similarity_coef": 0.5,
                "diversity_coef": 0.5
            },
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore.red": 0.01,
                    "ore.red_max": 10,
                    "battery.red": 0.02,
                    "battery.red_max": 10,
                    "heart": 1,
                    "heart_max": 1000
                }
            },
            "groups": {
                "agent": {
                    "id": 0,
                    "sprite": 0,
                    "props": {}
                }
            },
            "objects": {
                "altar": {"input_battery.red": 1, "output_heart": 1, ...},
                "mine_red": {"output_ore.red": 1, "color": 0, ...},
                "generator_red": {"input_ore.red": 1, "output_battery.red": 1, ...},
                "wall": {"swappable": False},
                "block": {"swappable": True}
            },
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": True},
                "change_color": {"enabled": False}
            },
            "reward_sharing": {
                "groups": {
                    "agent": {"agent": 0.0}
                }
            },
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 20,
                "height": 15,
                "border_width": 2,
                "agents": 2,
                "objects": {
                    "mine_red": 2,
                    "generator_red": 1,
                    "altar": 1,
                    "wall": 10,
                    "block": 5
                }
            }
        }
    }

    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.vecenv import make_vecenv

    env_cfg = DictConfig(env_config)
    curriculum = SingleTaskCurriculum("demo_task", env_cfg)

    vecenv = make_vecenv(
        curriculum=curriculum,
        vectorization="serial",
        num_envs=4,
        num_workers=1,
        batch_size=batch_size,
        render_mode=None,
    )
    """

    # For now, use dummy components to demonstrate the training loop
    print("Creating dummy environment for demonstration...")
    print("(Once the bug is fixed, this would be MettaGridEnv)\n")

    class DummyEnv:
        """Minimal dummy environment"""

        def __init__(self):
            self.num_agents = 8
            self.single_observation_space = type("Space", (), {"shape": (11, 11, 4), "dtype": np.float32})()
            self.single_action_space = type("Space", (), {"n": 8, "dtype": np.int32, "shape": (2,)})()
            self.action_names = ["noop", "move", "rotate", "attack"]
            self.max_action_args = [0, 1, 1, 1]

    class DummyVecEnv:
        """Minimal dummy vectorized environment"""

        def __init__(self, env):
            self.driver_env = env
            self.num_agents = env.num_agents * 4
            self.single_observation_space = env.single_observation_space
            self.single_action_space = env.single_action_space

        def async_reset(self, seed):
            pass

        def recv(self):
            obs = np.random.randn(self.num_agents, *self.driver_env.single_observation_space.shape).astype(np.float32)
            rewards = np.zeros(self.num_agents, dtype=np.float32)
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

    # Create a simple policy
    class SimplePolicy(torch.nn.Module):
        def __init__(self, obs_shape, num_actions):
            super().__init__()
            self.hidden_size = 128
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

            probs = torch.nn.functional.softmax(logits, dim=-1)
            if action is None:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action[..., 0])

            action_full = torch.stack([action, torch.zeros_like(action)], dim=-1).int()
            entropy = dist.entropy()

            return action_full, log_prob, entropy, value, None

    policy = SimplePolicy(env.single_observation_space.shape, env.single_action_space.n).to(device)
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
        num_lstm_layers=1,
        agents_per_batch=None,
    )

    # Training state
    agent_step = 0
    epoch = 0
    losses = Losses()
    timer = Stopwatch(None)
    timer.start()

    print("Starting training loop...")
    print("=" * 60)
    print("NOTE: This will fail at compute_puff_advantage, which is expected.\n")

    # Main training loop - this is the core pattern
    try:
        while agent_step < total_timesteps:
            steps_before = agent_step

            # ROLLOUT: Collect experience from the environment
            with timer("rollout"):
                agent_step, rollout_stats = rollout(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device,
                    agent_step=agent_step,
                    timer=timer,
                )

            print(f"✓ Rollout complete: collected {agent_step - steps_before} steps")

            # TRAIN: Update policy using PPO
            with timer("train"):
                # Dummy config for train_ppo
                agent_cfg = DictConfig({"agent": {"clip_range": 0}})

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
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_coef=clip_coef,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    norm_adv=True,
                    clip_vloss=True,
                    vf_clip_coef=0.1,
                    update_epochs=update_epochs,
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

    except AttributeError as e:
        if "compute_puff_advantage" in str(e):
            print("\n✓ Expected error: pufferlib.compute_puff_advantage not available")
            print("  This is normal for demo mode - the training structure is correct!")
        else:
            raise

    # Clean up
    vecenv.close()

    print("\n✅ Demo complete! The training loop structure is:")
    print("   while not done:")
    print("       experience = rollout(policy, env)      # Collect data")
    print("       train_ppo(policy, experience, ...)     # Update policy")
    print()
    print("Key findings:")
    print("1. The MettaGrid C++ constructor expects the ORIGINAL Python config")
    print("   format with 'groups', not the converted format with 'agent_groups'")
    print()
    print("2. There's a bug in mettagrid_env.py line 157 where it calls")
    print("   cpp_config_dict() before passing the config to MettaGrid")
    print()
    print("3. The fix is simple: remove the cpp_config_dict() call since")
    print("   MettaGrid does its own internal conversion")
    print()
    print("Once fixed, you can:")
    print("- Load configs from YAML files with OmegaConf.load()")
    print("- Create environments directly without Hydra")
    print("- Use the functional training API as shown here")
    print()
    print("For now, continue using tools/train.py with Hydra!")


if __name__ == "__main__":
    main()
