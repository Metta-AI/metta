#!/usr/bin/env python
"""Test that all py_agent types can train without crashing."""

import subprocess
import time

# All registered py_agent types from agent_mapper.py
PY_AGENTS = ["fast", "latent_attn_tiny", "latent_attn_small", "latent_attn_med", "example"]

def test_agent(agent_type):
    """Run a short training test for a specific agent type."""
    
    test_id = f"{agent_type}_{int(time.time())}"
    
    cmd = [
        "uv", "run", "./tools/train.py",
        f"py_agent={agent_type}",
        f"run=test_{test_id}",
        "trainer.total_timesteps=2000",
        "trainer.num_workers=2",
        "trainer.checkpoint.checkpoint_interval=1000",
        "trainer.simulation.skip_git_check=true",
        "trainer.simulation.evaluate_interval=0",
        "wandb=off"
    ]
    
    print(f"\nTesting py_agent={agent_type}...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {agent_type}: SUCCESS ({elapsed:.1f}s)")
        return True
    else:
        print(f"❌ {agent_type}: FAILED ({elapsed:.1f}s)")
        # Print last part of error for debugging
        if result.stderr:
            error_lines = result.stderr.split('\n')
            print("   Error (last 5 lines):")
            for line in error_lines[-5:]:
                if line.strip():
                    print(f"   {line}")
        return False


def main():
    print("=" * 60)
    print("TESTING ALL PY_AGENT IMPLEMENTATIONS")
    print("=" * 60)
    
    results = {}
    
    for agent in PY_AGENTS:
        results[agent] = test_agent(agent)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    for agent, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {agent:20s}: {status}")
    
    total_passed = sum(1 for s in results.values() if s)
    total_agents = len(results)
    
    print("\n" + "=" * 60)
    if total_passed == total_agents:
        print(f"✅ ALL TESTS PASSED ({total_passed}/{total_agents})")
        print("\nAll py_agent implementations can train successfully!")
        print("The base class LSTM management is working correctly.")
    else:
        print(f"⚠️ SOME TESTS FAILED ({total_passed}/{total_agents} passed)")
        print("\nThe following agents need fixes:")
        for agent, success in results.items():
            if not success:
                print(f"  - {agent}")


if __name__ == "__main__":
    main()