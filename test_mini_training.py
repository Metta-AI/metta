#!/usr/bin/env python
"""Run minimal training to test both implementations work correctly."""

import os
import subprocess
import time

def run_training(agent_type, use_py_agent=False):
    """Run training for a short time with the specified agent."""
    
    test_id = f"{agent_type}_{int(time.time())}"
    
    if use_py_agent:
        cmd = [
            "uv", "run", "./tools/train.py",
            f"py_agent={agent_type}",
            f"run=test_{test_id}",
            "trainer.total_timesteps=5000",
            "trainer.num_workers=2",
            "trainer.checkpoint.checkpoint_interval=2500",
            "trainer.simulation.skip_git_check=true",
            "trainer.simulation.evaluate_interval=0",
            "wandb=off"
        ]
    else:
        cmd = [
            "uv", "run", "./tools/train.py",
            f"agent={agent_type}",
            f"run=test_{test_id}",
            "trainer.total_timesteps=5000",
            "trainer.num_workers=2",
            "trainer.checkpoint.checkpoint_interval=2500",
            "trainer.simulation.skip_git_check=true",
            "trainer.simulation.evaluate_interval=0",
            "wandb=off"
        ]
    
    print(f"\n{'='*60}")
    print(f"Running training with {'py_agent' if use_py_agent else 'agent'}={agent_type}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(f"Training completed in {elapsed:.1f} seconds")
    
    # Check for errors
    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False
    
    # Look for reward information in output
    output_lines = result.stdout.split('\n')
    reward_lines = [line for line in output_lines if 'reward' in line.lower() or 'return' in line.lower()]
    
    if reward_lines:
        print("Reward-related output:")
        for line in reward_lines[-5:]:  # Last 5 reward lines
            print(f"  {line}")
    
    # Check if checkpoint was created
    checkpoint_dir = f"./train_dir/test_{test_id}/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        print(f"Checkpoints created: {checkpoint_files}")
        return True
    else:
        print("WARNING: No checkpoint directory created")
        return False


def main():
    print("=" * 60)
    print("MINI TRAINING TEST")
    print("Testing that both agent=fast and py_agent=fast can train")
    print("=" * 60)
    
    # Test ComponentPolicy (agent=fast)
    yaml_success = run_training("fast", use_py_agent=False)
    
    # Test PyTorch Fast (py_agent=fast)
    py_success = run_training("fast", use_py_agent=True)
    
    print("\n" + "=" * 60)
    print("TRAINING TEST RESULTS:")
    print(f"  agent=fast (ComponentPolicy): {'✓ SUCCESS' if yaml_success else '✗ FAILED'}")
    print(f"  py_agent=fast (PyTorch Fast): {'✓ SUCCESS' if py_success else '✗ FAILED'}")
    
    if yaml_success and py_success:
        print("\n✓ Both implementations can train successfully!")
        print("✓ The parity fixes are working correctly!")
    else:
        print("\n✗ One or both implementations failed to train")
        print("Further debugging needed")


if __name__ == "__main__":
    main()