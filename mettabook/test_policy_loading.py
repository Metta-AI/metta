#!/usr/bin/env python3
"""
Test script to troubleshoot policy loading issues.
"""

import sys
from pathlib import Path

# Add the analysis directory to the path
analysis_dir = Path(__file__).parent / "analysis"
sys.path.insert(0, str(analysis_dir.parent))

from analysis.observatory_client import ObservatoryClient  # noqa: E402


def main():
    print("🔍 Policy Loading Troubleshooting Script")
    print("=" * 50)

    # Initialize client
    client = ObservatoryClient()

    try:
        # Test basic connection
        print("\n1. Testing basic connection...")
        envs = client.get_environments()
        print(f"✅ Connected successfully. Found {len(envs)} environments.")

        # Run comprehensive debugging
        print("\n2. Running comprehensive database debugging...")
        client.debug_database_structure()

        # Test simple policy query
        print("\n3. Testing simple policy query...")
        simple_policies = client.test_simple_policy_query(n=5)

        if simple_policies:
            print(f"✅ Simple query works! Found {len(simple_policies)} policies.")
            print("Sample policies:")
            for i, policy in enumerate(simple_policies[:3]):
                print(f"   {i + 1}. {policy['name']} (ID: {policy['id']}) - {policy['episode_count']} episodes")
        else:
            print("❌ Simple query failed - no policies found.")

        # Test metric availability
        print("\n4. Testing metric availability...")
        reward_metrics = client.test_metric_query("reward")

        if reward_metrics:
            print("✅ 'reward' metric is available and has data.")
        else:
            print("❌ 'reward' metric not found or has no data.")

            # Try other common metric names
            alternative_metrics = ["return", "score", "performance", "value", "reward_sum"]
            for metric in alternative_metrics:
                print(f"\n   Trying alternative metric: '{metric}'")
                alt_metrics = client.test_metric_query(metric)
                if alt_metrics:
                    print(f"✅ Found metric '{metric}' with data!")
                    break

        # Test the actual get_top_policies method
        print("\n5. Testing get_top_policies method...")
        try:
            policies = client.get_top_policies(n=5, environments=None)
            if policies:
                print(f"✅ get_top_policies works! Found {len(policies)} policies.")
                for i, policy in enumerate(policies[:3]):
                    print(f"   {i + 1}. {policy['name']} - Avg reward: {policy['avg_reward']:.3f}")
            else:
                print("❌ get_top_policies returned no policies.")
        except Exception as e:
            print(f"❌ get_top_policies failed: {e}")

        # Summary
        print("\n" + "=" * 50)
        print("📋 TROUBLESHOOTING SUMMARY")
        print("=" * 50)

        if simple_policies:
            print("✅ Basic policy queries work")
        else:
            print("❌ Basic policy queries fail - check database connectivity")

        if reward_metrics:
            print("✅ 'reward' metric is available")
        else:
            print("❌ 'reward' metric not found - check metric names")

        if simple_policies and reward_metrics:
            print("✅ Database structure looks good")
            print("💡 Issue might be with complex query optimization")
        else:
            print("❌ Database structure issues detected")
            print("💡 Fix basic connectivity before complex queries")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
