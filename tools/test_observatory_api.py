#!/usr/bin/env -S uv run
"""
Test script to verify observatory API connection and basic functionality.
"""

import sys
from pathlib import Path

# Add the tools directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from observatory_policy_analysis import ObservatoryAPIClient


def test_api_connection():
    """Test basic API connection and functionality."""
    print("🔍 Testing Observatory API Connection...")

    try:
        # Create API client
        client = ObservatoryAPIClient()
        print("✅ Successfully created API client")

        # Test whoami endpoint
        print("🔍 Testing authentication...")
        whoami = client._make_request("GET", "/whoami")
        print(f"✅ Authenticated as: {whoami}")

        # Test listing tables
        print("🔍 Testing table listing...")
        tables = client.list_tables()
        print(f"✅ Found {len(tables)} tables:")
        for table in tables[:5]:  # Show first 5 tables
            print(f"   - {table['table_name']} ({table['column_count']} columns, {table['row_count']} rows)")
        if len(tables) > 5:
            print(f"   ... and {len(tables) - 5} more tables")

        # Test a simple query: total policies
        print("🔍 Testing total policies query...")
        result = client.execute_query("SELECT COUNT(*) as total_policies FROM policies")
        total_policies = result["rows"][0][0]
        print(f"✅ Total policies in database: {total_policies}")

        # Test total episodes
        print("🔍 Testing total episodes query...")
        result = client.execute_query("SELECT COUNT(*) as total_episodes FROM episodes")
        total_episodes = result["rows"][0][0]
        print(f"✅ Total episodes in database: {total_episodes}")

        # Test total agent metrics
        print("🔍 Testing total agent metrics query...")
        result = client.execute_query("SELECT COUNT(*) as total_agent_metrics FROM episode_agent_metrics")
        total_agent_metrics = result["rows"][0][0]
        print(f"✅ Total agent metrics in database: {total_agent_metrics}")

        print("\n🎉 All API tests passed!")
        return True

    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_policy_extraction():
    """Test extracting a small number of policies."""
    print("\n🔍 Testing Policy Extraction...")

    try:
        from observatory_policy_analysis import ObservatoryPolicyAnalyzer

        # Create API client and analyzer
        client = ObservatoryAPIClient()
        analyzer = ObservatoryPolicyAnalyzer(client)

        # Extract top 5 policies
        print("🔍 Extracting top 5 policies...")
        policies_df = analyzer.get_top_policies(5)

        print(f"✅ Successfully extracted {len(policies_df)} policies")
        print("📊 Policy summary:")
        print(
            f"   - Average reward range: {policies_df['avg_reward'].min():.3f} - {policies_df['avg_reward'].max():.3f}"
        )
        print(f"   - Average episode count: {policies_df['episode_count'].mean():.1f}")

        # Show first policy details
        if len(policies_df) > 0:
            first_policy = policies_df.iloc[0]
            print(f"   - Top policy: {first_policy['policy_url'][:50]}...")
            print(f"     Average reward: {first_policy['avg_reward']:.3f}")
            print(f"     Episodes: {first_policy['episode_count']}")

        print("🎉 Policy extraction test passed!")
        return True

    except Exception as e:
        print(f"❌ Policy extraction test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Observatory API Test Suite")
    print("=" * 50)

    # Test API connection
    api_ok = test_api_connection()

    if api_ok:
        # Test policy extraction
        extraction_ok = test_policy_extraction()

        if extraction_ok:
            print("\n🎉 All tests passed! The observatory API is working correctly.")
            print("\nYou can now run the full pipeline:")
            print("  ./tools/policy_analysis_pipeline.py --num-policies 10")
        else:
            print("\n❌ Policy extraction test failed.")
            sys.exit(1)
    else:
        print("\n❌ API connection test failed.")
        print("Please check your authentication token:")
        print("  python devops/observatory_login.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
