#!/usr/bin/env python3
"""Replay structure test: Verify replay format matches spec."""

import json
import zlib
import tempfile

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout


def main():
    print("=" * 60)
    print("TEST 4: Replay Structure Validation")
    print("=" * 60)

    cfg = MettaGridConfig.EmptyRoom(num_agents=2)
    cfg.game.max_steps = 5

    policy_env = PolicyEnvInterface.from_mg_cfg(cfg)
    multi_policy = RandomMultiAgentPolicy(policy_env)
    agent_policies = [multi_policy.agent_policy(i) for i in range(2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        replay_writer = ReplayLogWriter(tmpdir)

        print("\n🏃 Running rollout...")
        rollout = Rollout(cfg, agent_policies, seed=42, event_handlers=[replay_writer])
        rollout.run_until_done()

        replay_path = replay_writer.get_written_replay_paths()[0]
        print(f"📄 Reading replay: {replay_path}")

        with open(replay_path, 'rb') as f:
            replay_data = json.loads(zlib.decompress(f.read()))

        print("\n📋 Validating replay structure...")

        # Check top-level keys
        required_keys = ['version', 'action_names', 'item_names', 'type_names',
                        'map_size', 'num_agents', 'max_steps', 'mg_config',
                        'policies', 'objects']

        for key in required_keys:
            assert key in replay_data, f"❌ FAIL: Missing required key '{key}'"
            print(f"   ✓ Has '{key}'")

        print("✅ PASS: All required top-level keys present")

        # Validate policies array structure
        print("\n📋 Validating policies array structure...")
        policies = replay_data['policies']
        assert isinstance(policies, list), "❌ FAIL: 'policies' must be a list"
        assert len(policies) > 0, "❌ FAIL: 'policies' array is empty"

        for i, policy in enumerate(policies):
            assert 'name' in policy, f"❌ FAIL: Policy {i} missing 'name'"
            assert 'uri' in policy, f"❌ FAIL: Policy {i} missing 'uri'"
            assert 'is_scripted' in policy, f"❌ FAIL: Policy {i} missing 'is_scripted'"
            assert isinstance(policy['name'], str), f"❌ FAIL: Policy {i} 'name' must be string"
            assert isinstance(policy['uri'], str), f"❌ FAIL: Policy {i} 'uri' must be string"
            assert isinstance(policy['is_scripted'], bool), f"❌ FAIL: Policy {i} 'is_scripted' must be bool"
            print(f"   ✓ Policy {i}: name='{policy['name']}', uri='{policy['uri']}', is_scripted={policy['is_scripted']}")

        print("✅ PASS: Policies array structure valid")

        # Validate agent objects have policy_id
        print("\n🤖 Validating agent objects...")
        agent_count = 0
        for obj in replay_data['objects']:
            if 'agent_id' in obj:
                agent_id = obj['agent_id']
                assert 'policy_id' in obj, f"❌ FAIL: Agent {agent_id} missing 'policy_id'"

                policy_id = obj['policy_id']
                # policy_id can be a constant or time series
                if isinstance(policy_id, list):
                    # Time series format [[step, value], ...]
                    assert len(policy_id) > 0, f"❌ FAIL: Agent {agent_id} policy_id time series is empty"
                    policy_id_value = policy_id[0][1] if isinstance(policy_id[0], list) else policy_id
                else:
                    policy_id_value = policy_id

                assert isinstance(policy_id_value, int), f"❌ FAIL: policy_id must be int"
                assert 0 <= policy_id_value < len(policies), f"❌ FAIL: policy_id {policy_id_value} out of range"

                policy_name = policies[policy_id_value]['name']
                print(f"   ✓ Agent {agent_id}: policy_id={policy_id_value} ({policy_name})")
                agent_count += 1

        assert agent_count == 2, f"❌ FAIL: Expected 2 agents, found {agent_count}"
        print("✅ PASS: All agent objects have valid policy_id")

        # Pretty print final structure
        print("\n📊 Replay structure summary:")
        print(f"   Version: {replay_data['version']}")
        print(f"   Agents: {replay_data['num_agents']}")
        print(f"   Max steps: {replay_data['max_steps']}")
        print(f"   Policies: {len(replay_data['policies'])}")
        print(f"   Objects: {len(replay_data['objects'])}")

        print("\n" + "=" * 60)
        print("✅ TEST 4 PASSED: Replay Structure")
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

