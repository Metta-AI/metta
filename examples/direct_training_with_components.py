"""Example of using Metta's modular training components directly.

This example shows how to build a custom training loop using the
individual components without the full MettaTrainer orchestrator.
"""

import torch

from metta.agent import SimpleCNNAgent
from metta.rl import (
    Experience,
    PPOOptimizer,
    RolloutCollector,
    make_vecenv,
)
from metta.rl.stats_logger import StatsLogger
from mettagrid.curriculum import StaticCurriculum


def custom_training_loop():
    """Example of building a custom training loop with modular components."""

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 32
    batch_size = 2048
    minibatch_size = 256
    total_timesteps = 1_000_000

    # Create environment
    env_config = {
        "width": 15,
        "height": 15,
        "num_agents": 4,
        "max_steps": 256,
    }

    # Create vectorized environment
    curriculum = StaticCurriculum(env_config)
    vecenv = make_vecenv(
        curriculum=curriculum,
        vectorization="serial",
        num_envs=num_envs,
        batch_size=num_envs,
        num_workers=1,
    )

    # Create agent
    agent = SimpleCNNAgent(
        obs_space=vecenv.single_observation_space,
        action_space=vecenv.single_action_space,
        obs_width=15,
        obs_height=15,
        hidden_size=256,
    ).to(device)

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=batch_size,
        bptt_horizon=16,
        minibatch_size=minibatch_size,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=256,
    )

    # Create training components
    collector = RolloutCollector(
        vecenv=vecenv,
        policy=agent,
        experience_buffer=experience,
        device=device,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    ppo_trainer = PPOOptimizer(
        policy=agent,
        optimizer=optimizer,
        device=device,
        clip_coef=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    stats_logger = StatsLogger()

    # Training loop
    agent_steps = 0
    epoch = 0

    print("Starting custom training loop...")

    while agent_steps < total_timesteps:
        # Collect rollouts
        print(f"\nEpoch {epoch} - Collecting rollouts...")
        stats, steps_collected = collector.collect()
        agent_steps = collector.agent_steps

        # Log environment stats
        stats_logger.add_stats(stats)

        # Update policy
        print(f"Training on {steps_collected} steps...")
        loss_stats = ppo_trainer.update(
            experience=experience,
            update_epochs=4,
        )

        # Print progress
        print(f"Agent steps: {agent_steps:,} / {total_timesteps:,}")
        print(
            f"Losses: policy={loss_stats['policy_loss']:.3f}, "
            f"value={loss_stats['value_loss']:.3f}, "
            f"entropy={loss_stats['entropy']:.3f}"
        )

        if "episode/reward" in stats:
            avg_reward = sum(stats["episode/reward"]) / len(stats["episode/reward"])
            print(f"Average episode reward: {avg_reward:.2f}")

        epoch += 1

    print("\nTraining complete!")

    # Save the trained model
    torch.save(agent.state_dict(), "custom_trained_agent.pt")
    print("Model saved to custom_trained_agent.pt")

    # Cleanup
    vecenv.close()


def evaluation_example():
    """Example of using the PolicyEvaluator component."""
    from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple evaluation suite
    sim_configs = {
        "test/navigation": SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=10,
            env_overrides={"width": 10, "height": 10},
        ),
        "test/cooperation": SingleEnvSimulationConfig(
            env="/env/mettagrid/mettagrid",
            num_episodes=10,
            env_overrides={"width": 15, "height": 15, "num_agents": 4},
        ),
    }

    sim_suite_config = SimulationSuiteConfig(simulations=sim_configs)

    # Create evaluator (would need policy_store in real usage)
    # evaluator = PolicyEvaluator(
    #     sim_suite_config=sim_suite_config,
    #     policy_store=policy_store,  # Would need actual policy store
    #     device=device,
    # )

    # Evaluate a policy
    # results = evaluator.evaluate(policy_record)
    # print(f"Evaluation results: {results}")


if __name__ == "__main__":
    # Run the custom training loop example
    custom_training_loop()

    # The evaluation example is commented out as it needs additional setup
    # evaluation_example()
