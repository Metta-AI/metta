"""Example usage of external PyTorch policies in Metta.

This script demonstrates how to:
1. Load an external PyTorch policy
2. Use it for evaluation
3. Continue training from an external checkpoint
"""

from omegaconf import OmegaConf

from metta.agent.external.pytorch_adapter import load_pytorch_policy


def evaluate_external_policy():
    """Example: Evaluate an external PyTorch policy."""

    # Configuration for external policy
    pytorch_cfg = OmegaConf.create(
        {"_target_": "metta.agent.external.torch.Recurrent", "hidden_size": 512, "cnn_channels": 128}
    )

    # Load the external policy
    checkpoint_path = "checkpoints/metta_6-8/metta_6-8.pt"
    policy = load_pytorch_policy(checkpoint_path, device="cuda", pytorch_cfg=pytorch_cfg)

    # Create environment config
    env_cfg = OmegaConf.create(
        {
            "mettagrid": {
                "observation_space": "tokenized",
                "max_observation_tokens": 200,
                "groups": [{"name": "agent", "num_agents": 1}],
            }
        }
    )

    # Run evaluation
    print(f"Evaluating external policy from: {checkpoint_path}")
    # ... evaluation code ...


def continue_training_from_external():
    """Example: Continue training from an external checkpoint."""

    from metta.train.trainer import MettaTrainer

    # Full configuration
    cfg = OmegaConf.create(
        {
            "policy_uri": "pytorch://checkpoints/external_model.pt",
            "pytorch": {
                "_target_": "metta.agent.external.lstm_transformer.Recurrent",
                "hidden_size": 384,
                "depth": 3,
                "num_heads": 6,
            },
            "trainer": {"total_timesteps": 10_000_000, "batch_size": 128, "learning_rate": 3e-4},
            # ... other config ...
        }
    )

    # The trainer will automatically load the external policy
    # and continue training from that checkpoint
    trainer = MettaTrainer(cfg)
    trainer.train()


def compare_policies():
    """Example: Compare external vs native policies."""

    # Load external policy
    external_policy = load_pytorch_policy(
        "checkpoints/pufferlib_model.pt", pytorch_cfg={"_target_": "metta.agent.external.torch.Recurrent"}
    )

    # Load native Metta policy
    from metta.agent.policy_store import PolicyStore

    store = PolicyStore(cfg={}, wandb_run=None)
    native_pr = store.load_from_uri("file://checkpoints/metta_native.pt")

    # Compare performance
    # ... comparison code ...


if __name__ == "__main__":
    print("External Policy Usage Examples")
    print("1. Evaluating external policy...")
    evaluate_external_policy()

    print("\n2. Continuing training from external checkpoint...")
    # continue_training_from_external()  # Uncomment to run

    print("\n3. Comparing policies...")
    # compare_policies()  # Uncomment to run
