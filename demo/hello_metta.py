#!/usr/bin/env -S uv run

"""Hello Metta demo - demonstrates how to run training with the new config system."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add the project root to Python path so we can import from tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from metta.rl.trainer_config import TrainerConfig
from tools.train import TrainConfig


def create_demo_train_config() -> TrainConfig:
    """Create a demo training configuration."""

    # Create a TrainerConfig with minimal settings for fast demo
    run_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_dir = f"./train_dir/{run_id}"
    
    trainer_config = TrainerConfig(
        total_timesteps=10,  # Very small for demo
        num_workers=1,

        # PPO settings - using defaults but with small batch size
        batch_size=512,
        minibatch_size=64,

        # Checkpoint settings - must provide valid paths
        checkpoint={"checkpoint_interval": 5, "checkpoint_dir": f"{train_dir}/checkpoints", "wandb_checkpoint_interval": 0},

        # Simulation settings - must provide valid replay_dir
        simulation={"evaluate_interval": 0, "replay_dir": f"{train_dir}/replays", "evaluate_remote": False, "evaluate_local": False}
    )

    # Create the main train config
    train_config = TrainConfig(
        run=run_id,
        trainer=trainer_config,
        device="cpu",
        wandb={"enabled": False}  # Disable wandb for demo
    )

    return train_config


def main():
    """Main demo function."""
    print("🎯 Hello Metta Demo - Training Configuration Example")
    print("=" * 50)

    # Create demo config
    print("📝 Creating demo training configuration...")
    config = create_demo_train_config()
    run_id = config.run

    print(f"   Run ID: {run_id}")
    print(f"   Total timesteps: {config.trainer.total_timesteps}")
    print(f"   Device: {config.device}")
    print(f"   Num workers: {config.trainer.num_workers}")

    # Create train_dir structure
    train_dir = Path("train_dir") / run_id
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save config to YAML
    config_path = train_dir / "train_cfg.yaml"
    print(f"💾 Saving configuration to: {config_path}")

    # Convert to dict and save
    config_dict = config.model_dump()
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print("✅ Configuration saved successfully!")
    print()

    # Show the config content
    print("📋 Configuration content:")
    print("-" * 30)
    with open(config_path, 'r') as f:
        config_content = f.read()
        # Show just the key parts to keep output manageable
        lines = config_content.split('\n')[:20]  # First 20 lines
        print('\n'.join(lines))
        if len(config_content.split('\n')) > 20:
            print("  ... (truncated)")
    print()

    # Demonstrate how to run training
    print("🚀 To run training, execute:")
    print(f"   python tools/train.py --config {config_path}")
    print()

    # Actually run the training for demo
    print("🎬 Running training demo...")
    print("-" * 30)

    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent

        # Run training subprocess
        result = subprocess.run(
            [sys.executable, "tools/train.py", "--config", str(config_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for demo
        )

        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\nTraining output (last 10 lines):")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                print(f"   {line}")
        else:
            print("❌ Training failed!")
            print("Error output:")
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"   {line}")

    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (this is normal for demo)")
    except Exception as e:
        print(f"❌ Error running training: {e}")

    print()
    print("📁 Output files:")
    if train_dir.exists():
        for item in train_dir.rglob('*'):
            if item.is_file():
                print(f"   {item}")

    print()
    print("🎉 Demo completed!")
    print("   You can now modify the config YAML and run training with different settings.")
    print(f"   Config file: {config_path}")
    print(f"   Run: python tools/train.py --config {config_path}")


if __name__ == "__main__":
    main()
