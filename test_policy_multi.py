#!/usr/bin/env python3
"""Multi-policy test: Different policies controlling different agents."""

import json
import zlib
import tempfile

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.policy.noop import NoopPolicy
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout


def main():
    print("=" * 60)
    print("TEST 2: Multi-Policy (Random + Noop)")
    print("=" * 60)

    cfg = MettaGridConfig.EmptyRoom(num_agents=4)
    cfg.game.max_steps = 10

    policy_env = PolicyEnvInterface.from_mg_cfg(cfg)

    # Create two different policies
    random_policy = RandomMultiAgentPolicy(policy_env)
    noop_policy = NoopPolicy(policy_env)

    # Assign different policies to different agents
    agent_policies = [
        random_policy.agent_policy(0),  # Agent 0: random
        random_policy.agent_policy(1),  # Agent 1: random
        noop_policy.agent_policy(2),    # Agent 2: noop
        noop_policy.agent_policy(3),    # Agent 3: noop
    ]

    print("\n📋 Policy assignments:")
    print("   Agents 0-1: random policy")
    print("   Agents 2-3: noop policy")

    with tempfile.TemporaryDirectory() as tmpdir:
        replay_writer = ReplayLogWriter(tmpdir)

        print("\n🏃 Running rollout...")
        rollout = Rollout(cfg, agent_policies, seed=42, event_handlers=[replay_writer])
        rollout.run_until_done()

        replay_path = replay_writer.get_written_replay_paths()[0]
        print(f"📄 Reading replay from: {replay_path}")

        with open(replay_path, 'rb') as f:
            replay_data = json.loads(zlib.decompress(f.read()))

        # Verify policies array
        policies = replay_data['policies']
        assert len(policies) == 2, f"❌ FAIL: Expected 2 policies, found {len(policies)}"
        print(f"✅ PASS: Found 2 unique policies")

        # Verify policy names
        policy_names = {p['name'] for p in policies}
        assert 'random' in policy_names, "❌ FAIL: Missing 'random' policy"
        assert 'noop' in policy_names, "❌ FAIL: Missing 'noop' policy"
        print(f"✅ PASS: Both 'random' and 'noop' policies present")

        # Create name->id mapping
        name_to_id = {p['name']: i for i, p in enumerate(policies)}

        # Verify agent assignments
        print("\n🤖 Verifying agent policy assignments:")
        agent_policy_map = {}
        for obj in replay_data['objects']:
            if 'agent_id' in obj:
                agent_id = obj['agent_id']
                policy_id = obj.get('policy_id')
                assert policy_id is not None, f"❌ FAIL: Agent {agent_id} missing policy_id"

                policy_name = policies[policy_id]['name']
                agent_policy_map[agent_id] = policy_name
                print(f"   Agent {agent_id}: policy_id={policy_id} ({policy_name})")

        # Verify correct assignments
        assert agent_policy_map[0] == 'random', f"❌ FAIL: Agent 0 should be 'random'"
        assert agent_policy_map[1] == 'random', f"❌ FAIL: Agent 1 should be 'random'"
        assert agent_policy_map[2] == 'noop', f"❌ FAIL: Agent 2 should be 'noop'"
        assert agent_policy_map[3] == 'noop', f"❌ FAIL: Agent 3 should be 'noop'"
        print("✅ PASS: All agents assigned to correct policies")

        print("\n" + "=" * 60)
        print("✅ TEST 2 PASSED: Multi-Policy")
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

