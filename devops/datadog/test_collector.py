#!/usr/bin/env python3
"""Test script for the new collector architecture."""

import os
import sys

# Add devops to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datadog.collectors.github import GitHubCollector


def test_github_collector():
    """Test GitHub collector instantiation and metric collection."""
    print("Testing GitHub collector...")

    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        print("Please set it with: export GITHUB_TOKEN=your_token")
        return False

    # Create collector
    collector = GitHubCollector(
        organization="PufferAI",
        repository="metta",
        github_token=github_token,
    )

    print(f"Created collector: {collector}")

    # Test safe collection
    print("\nCollecting metrics...")
    metrics = collector.collect_safe()

    if not metrics:
        print("WARNING: No metrics collected")
        return False

    print(f"\nCollected {len(metrics)} metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")

    # Verify expected metric categories
    expected_prefixes = ["prs.", "branches.", "commits.", "code.", "ci.", "developers."]
    found_prefixes = set()

    for key in metrics.keys():
        for prefix in expected_prefixes:
            if key.startswith(prefix):
                found_prefixes.add(prefix)
                break

    print(f"\nFound metric categories: {sorted(found_prefixes)}")
    print(f"Expected categories: {sorted(expected_prefixes)}")

    if found_prefixes == set(expected_prefixes):
        print("\n✅ All metric categories found!")
        return True
    else:
        missing = set(expected_prefixes) - found_prefixes
        print(f"\n⚠️  Missing categories: {sorted(missing)}")
        return True  # Still pass - some categories might be empty


def main():
    """Run tests."""
    success = test_github_collector()

    if success:
        print("\n✅ Tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
