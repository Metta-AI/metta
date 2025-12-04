#!/usr/bin/env python3
"""Training environment test: Verify training policy descriptor."""

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator


def main():
    print("=" * 60)
    print("TEST 3: Training Environment Policy Descriptor")
    print("=" * 60)

    cfg = MettaGridConfig.EmptyRoom(num_agents=4)
    sim = Simulator()

    print("\n🏋️ Creating training environment...")
    env = MettaGridPufferEnv(sim, cfg)

    # Check simulation has policy descriptors
    descriptors = env.current_simulation.get_policy_descriptors()

    assert len(descriptors) == 4, f"❌ FAIL: Expected 4 descriptors, got {len(descriptors)}"
    print("✅ PASS: Environment has 4 policy descriptors (one per agent)")

    # Verify all are training policy
    for i, desc in enumerate(descriptors):
        assert desc.name == "training", f"❌ FAIL: Agent {i} descriptor name is '{desc.name}', expected 'training'"
        assert desc.is_scripted is False, "❌ FAIL: Training policy should not be scripted"
        print(f"   Agent {i}: name='{desc.name}', is_scripted={desc.is_scripted}")

    print("✅ PASS: All agents use 'training' policy descriptor")

    # Verify unique policies
    unique = env.current_simulation.get_unique_policy_descriptors()
    assert len(unique) == 1, f"❌ FAIL: Expected 1 unique policy, got {len(unique)}"
    assert unique[0].name == "training", "❌ FAIL: Unique policy should be 'training'"
    print("✅ PASS: Only 1 unique 'training' policy")

    # Verify policy IDs
    for agent_id in range(4):
        policy_id = env.current_simulation.get_agent_policy_id(agent_id)
        assert policy_id == 0, f"❌ FAIL: Agent {agent_id} has policy_id {policy_id}, expected 0"
    print("✅ PASS: All agents have policy_id=0")

    env.close()

    print("\n" + "=" * 60)
    print("✅ TEST 3 PASSED: Training Environment")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

