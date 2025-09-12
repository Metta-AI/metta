#!/usr/bin/env python3
"""
Test the full environment grid visualization
"""

import sys
from pathlib import Path

# Add metta to path
sys.path.insert(0, str(Path(__file__).parent))


def test_full_grid():
    print("🌍 Testing Full Environment Grid Visualization")
    print("=" * 60)

    # Import the tribal play tool
    from experiments.recipes.tribal_basic import TribalHeadlessPlayTool, make_tribal_environment

    # Create the play tool with shorter episode for testing
    env_config = make_tribal_environment(max_steps=50)
    play_tool = TribalHeadlessPlayTool(env_config=env_config, policy_uri="test_move")

    print("✅ Tribal play tool created")
    print("🎮 Running with full environment grid...")

    # Run the tool
    result = play_tool.invoke({}, [])

    return result == 0


if __name__ == "__main__":
    try:
        success = test_full_grid()
        if success:
            print("\n🎉 SUCCESS: Full environment grid visualization working!")
        else:
            print("\n❌ FAILED: Test failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
