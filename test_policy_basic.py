#!/usr/bin/env python3
"""Basic test: Single policy controlling all agents."""

import json
import zlib
import tempfile
from pathlib import Path

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout


def main():
    print("=" * 60)
    print("TEST 1: Basic Single Policy")
    print("=" * 60)

    # Create config
    cfg = MettaGridConfig.EmptyRoom(num_agents=4)
    cfg.game.max_steps = 10

    # Create policy
    policy_env = PolicyEnvInterface.from_mg_cfg(cfg)
    multi_policy = RandomMultiAgentPolicy(policy_env)
    agent_policies = [multi_policy.agent_policy(i) for i in range(4)]

    # Create replay writer
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_writer = ReplayLogWriter(tmpdir)

        # Run rollout
        print("\n🏃 Running rollout...")
        rollout = Rollout(cfg, agent_policies, seed=42, event_handlers=[replay_writer])
        rollout.run_until_done()

        # Get replay path
        replay_paths = replay_writer.get_written_replay_paths()
        replay_path = replay_paths[0]

        # Read and inspect replay
        print(f"📄 Reading replay from: {replay_path}")
        with open(replay_path, 'rb') as f:
            compressed = f.read()
        decompressed = zlib.decompress(compressed)
        replay_data = json.loads(decompressed)

        # Verify policies array exists
        assert 'policies' in replay_data, "❌ FAIL: 'policies' key missing from replay"
        print("✅ PASS: Replay contains 'policies' array")

        # Verify policy structure
        policies = replay_data['policies']
        assert len(policies) == 1, f"❌ FAIL: Expected 1 policy, found {len(policies)}"
        print(f"✅ PASS: Exactly 1 unique policy found")

        policy = policies[0]
        assert policy['name'] == 'random', f"❌ FAIL: Expected 'random', got '{policy['name']}'"
        assert 'uri' in policy, "❌ FAIL: Policy missing 'uri' field"
        assert 'is_scripted' in policy, "❌ FAIL: Policy missing 'is_scripted' field"
        assert policy['is_scripted'] is True, "❌ FAIL: Random policy should be scripted"
        print(f"✅ PASS: Policy structure correct: {policy}")

        # Verify agents have policy_id
        agent_count = 0
        for obj in replay_data['objects']:
            if 'agent_id' in obj:
                agent_id = obj['agent_id']
                policy_id = obj.get('policy_id')
                assert policy_id is not None, f"❌ FAIL: Agent {agent_id} missing 'policy_id'"
                assert policy_id == 0, f"❌ FAIL: Agent {agent_id} has wrong policy_id: {policy_id}"
                agent_count += 1

        assert agent_count == 4, f"❌ FAIL: Expected 4 agents, found {agent_count}"
        print(f"✅ PASS: All 4 agents have correct policy_id=0")

        print("\n" + "=" * 60)
        print("✅ TEST 1 PASSED: Basic Single Policy")
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

