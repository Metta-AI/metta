#!/usr/bin/env python3
"""
Simplified Functional Training Demo

This demonstrates the core concepts of functional training in Metta
without the complexity of full environment setup.

Key points:
1. Direct object creation (no Hydra)
2. Pydantic configs for structured configuration
3. Simple training loop with rollout() and train_ppo()
4. Clear separation of concerns

For a complete working example with real environments, see demo.py
"""

from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel


# Pydantic Configs for structured configuration
class PPOConfig(BaseModel):
    """PPO algorithm parameters"""

    gamma: float = 0.977
    gae_lambda: float = 0.916
    clip_coef: float = 0.1
    ent_coef: float = 0.0021
    vf_coef: float = 0.44
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_clip_coef: float = 0.1
    target_kl: Optional[float] = None


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    type: str = "adam"
    learning_rate: float = 0.0004573146765703167
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-12
    weight_decay: float = 0.0


class TrainingConfig(BaseModel):
    """Training loop configuration"""

    total_timesteps: int = 100_000
    batch_size: int = 2048
    minibatch_size: int = 256
    bptt_horizon: int = 64
    update_epochs: int = 4
    checkpoint_interval: int = 10
    device: str = "cpu"


def create_optimizer(config: OptimizerConfig, policy: torch.nn.Module):
    """Create optimizer from config"""
    if config.type == "adam":
        return torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")


def functional_training_loop():
    """
    Demonstrates the core functional training pattern in Metta.

    This is a conceptual example showing how the training loop works
    without the complexity of actual environment setup.
    """

    print("🚀 Simplified Functional Training Demo")
    print("=" * 50)
    print("\nThis demo shows the STRUCTURE of functional training.")
    print("For a complete working example, see demo.py\n")

    # 1. Create configurations using Pydantic
    ppo_config = PPOConfig()
    optimizer_config = OptimizerConfig()
    training_config = TrainingConfig(
        total_timesteps=1000,  # Small for demo
        checkpoint_interval=100,
    )

    print("1. Created Pydantic configurations:")
    print(f"   - PPO: gamma={ppo_config.gamma}, clip_coef={ppo_config.clip_coef}")
    print(f"   - Optimizer: {optimizer_config.type}, lr={optimizer_config.learning_rate}")
    print(f"   - Training: {training_config.total_timesteps} steps\n")

    # 2. Create objects directly (no Hydra magic)
    device = torch.device(training_config.device)

    # Mock policy for demonstration
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

        def forward(self, obs, state=None, action=None):
            # Mock forward pass
            batch_size = obs.shape[0]
            actions = torch.randint(0, 10, (batch_size, 2))
            logprobs = torch.randn(batch_size)
            entropy = torch.randn(batch_size)
            values = torch.randn(batch_size)
            full_logprobs = torch.randn(batch_size, 10)
            return actions, logprobs, entropy, values, full_logprobs

    policy = MockPolicy().to(device)
    optimizer = create_optimizer(optimizer_config, policy)

    print("2. Created objects directly:")
    print(f"   - Policy: {policy.__class__.__name__}")
    print(f"   - Optimizer: {optimizer.__class__.__name__}")
    print(f"   - Device: {device}\n")

    # 3. Initialize training state
    agent_step = 0
    epoch = 0

    print("3. Starting functional training loop...")
    print("   (Using mock data for demonstration)\n")

    # 4. Main training loop - the key pattern!
    while agent_step < training_config.total_timesteps:
        # ========== ROLLOUT PHASE ==========
        print(f"Epoch {epoch}: Rollout phase")

        # In real code, this would be:
        # agent_step, stats = rollout(
        #     policy, vecenv, experience, device, agent_step, timer
        # )

        # Mock rollout
        rollout_steps = 128
        agent_step += rollout_steps

        # ========== TRAINING PHASE ==========
        print(f"Epoch {epoch}: Training phase")

        # In real code, this would be:
        # epoch = train_ppo(
        #     policy=policy,
        #     optimizer=optimizer,
        #     experience=experience,
        #     device=device,
        #     losses=losses,
        #     epoch=epoch,
        #     cfg=cfg,
        #     **ppo_config.dict(),
        # )

        # Mock training
        for _ in range(training_config.update_epochs):
            # Mock loss computation
            loss = torch.randn(1, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.max_grad_norm)
            optimizer.step()

        epoch += 1

        # ========== LOGGING ==========
        print(f"   - Steps: {agent_step}/{training_config.total_timesteps}")
        print(f"   - Mock SPS: {np.random.randint(1000, 2000)}")
        print()

        # ========== CHECKPOINTING ==========
        if epoch % training_config.checkpoint_interval == 0:
            print(f"💾 Saving checkpoint at epoch {epoch}")
            # In real code:
            # policy_store.save(name, path, policy, metadata)
            print()

    print("✅ Training complete!")
    print("\nKey takeaways:")
    print("1. All objects created explicitly (no magic)")
    print("2. Configuration via Pydantic models")
    print("3. Simple while loop with rollout() and train_ppo()")
    print("4. Full control over the training process")
    print("\nFor a complete implementation, see:")
    print("- demo.py (full working example)")
    print("- metta/rl/functional_trainer.py (rollout & train_ppo)")


# Example of how you might structure a custom training loop
def custom_training_example():
    """Shows how to create a completely custom training loop"""
    print("\n" + "=" * 50)
    print("Custom Training Loop Example")
    print("=" * 50 + "\n")

    # Your custom setup
    policy = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(policy.parameters())

    print("With functional training, you can:")
    print("1. Use any optimizer (not just Adam/Muon)")
    print("2. Implement custom advantage computation")
    print("3. Add your own losses")
    print("4. Control exactly when checkpoints happen")
    print("5. Integrate with any external tools")
    print("\nExample custom loop structure:\n")

    print("""
    for epoch in range(1000):
        # Custom rollout
        data = my_custom_rollout(policy, env)

        # Custom advantage computation
        advantages = my_special_advantage_function(data)

        # Custom losses
        policy_loss = compute_policy_loss(data, advantages)
        my_custom_loss = compute_auxiliary_loss(data)

        # Update
        total_loss = policy_loss + my_custom_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Custom logging
        log_to_my_system({"epoch": epoch, "loss": total_loss})
    """)


if __name__ == "__main__":
    # Run the simplified demo
    functional_training_loop()

    # Show custom training example
    custom_training_example()

    print("\n📚 Next steps:")
    print("1. Check demo.py for a full working example")
    print("2. Read FUNCTIONAL_TRAINING.md for detailed docs")
    print("3. Explore metta/rl/functional_trainer.py for implementation")
