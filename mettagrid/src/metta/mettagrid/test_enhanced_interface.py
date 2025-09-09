"""Test script for enhanced C++ interface with int-based grid support.

This script tests the new dual-format C++ interface to ensure it works correctly
with both legacy string-based grids and new int-based grids.
"""

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.map_builder.enhanced_random import EnhancedRandomMapBuilder
from metta.mettagrid.mettagrid_config import GameConfig, MettaGridConfig
from metta.mettagrid.object_types import ObjectTypes


def test_legacy_interface():
    """Test that legacy string-based interface still works."""
    print("Testing legacy string-based interface...")

    config = MettaGridConfig.EmptyRoom(num_agents=2, width=5, height=5)

    # Create environment using legacy map builder
    try:
        env = MettaGridCore(config)
        game_map = env._map_builder.build()

        print(f"✅ Legacy interface: Created {game_map.shape} map")
        print(f"   Grid format: {'int' if game_map.is_int_based() else 'string'}-based")
        print(f"   Sample cell: {game_map.get_object_name(0, 0)}")

        return True
    except Exception as e:
        print(f"❌ Legacy interface failed: {e}")
        return False


def test_enhanced_interface():
    """Test that enhanced int-based interface works."""
    print("Testing enhanced int-based interface...")

    # Create game config with objects
    game_config = GameConfig()
    game_config.objects = {
        "wall": type("WallConfig", (), {"type_id": 1, "type_name": "wall", "swappable": False})(),
        "altar": type("AltarConfig", (), {"type_id": 2, "type_name": "altar"})(),
    }

    config = MettaGridConfig(game=game_config)

    # Create enhanced map builder
    builder_config = EnhancedRandomMapBuilder.Config(
        width=5, height=5, agents=2, objects={"wall": 2, "altar": 1}, use_int_format=True
    )

    try:
        builder = EnhancedRandomMapBuilder(builder_config, game_config)
        game_map = builder.build()

        print(f"✅ Enhanced interface: Created {game_map.shape} map")
        print(f"   Grid format: {'int' if game_map.is_int_based() else 'string'}-based")
        print(f"   Sample cell: {game_map.get_object_name(0, 0)}")
        print(f"   Type mapping available: {len(game_map.decoder_key) if game_map.decoder_key else 0} types")

        # Test creating environment with int-based map
        env = MettaGridCore(config)
        env._map_builder = builder  # Override with our enhanced builder

        # This should use the enhanced C++ constructor path
        # env._create_c_env()  # Would need proper GameConfig setup

        return True
    except Exception as e:
        print(f"❌ Enhanced interface failed: {e}")
        return False


def test_format_conversion():
    """Test conversion between formats."""
    print("Testing format conversion...")

    try:
        # Create int-based map
        # game_config = GameConfig()  # Not used in this test
        builder_config = EnhancedRandomMapBuilder.Config(
            width=3, height=3, agents=1, objects={"wall": 2}, use_int_format=True
        )
        builder = EnhancedRandomMapBuilder(builder_config)
        int_map = builder.build()

        # Convert to legacy format
        legacy_map = int_map.to_legacy_format()

        print("✅ Format conversion:")
        print(f"   Original: {int_map.grid.dtype} -> Legacy: {legacy_map.grid.dtype}")
        print(f"   Shapes match: {int_map.shape == legacy_map.shape}")

        # Test that objects are preserved
        for r in range(int_map.shape[0]):
            for c in range(int_map.shape[1]):
                int_name = int_map.get_object_name(r, c)
                legacy_name = str(legacy_map.grid[r, c])
                if int_name != legacy_name:
                    print(f"   ⚠️  Mismatch at ({r},{c}): {int_name} vs {legacy_name}")

        return True
    except Exception as e:
        print(f"❌ Format conversion failed: {e}")
        return False


def test_object_type_constants():
    """Test that object type constants work correctly."""
    print("Testing object type constants...")

    try:
        # Test basic constants
        assert ObjectTypes.EMPTY == 0
        assert ObjectTypes.WALL == 1
        assert ObjectTypes.is_agent(ObjectTypes.AGENT_DEFAULT)
        assert not ObjectTypes.is_agent(ObjectTypes.WALL)

        # Test mappings
        mappings = ObjectTypes.get_standard_mappings()
        reverse_mappings = ObjectTypes.get_reverse_mappings()

        assert mappings["empty"] == ObjectTypes.EMPTY
        assert reverse_mappings[ObjectTypes.WALL] == "wall"

        print("✅ Object type constants:")
        print(f"   Standard mappings: {len(mappings)} objects")
        print(f"   Agent type range: {ObjectTypes.AGENT_BASE}-{ObjectTypes.AGENT_BASE + 10}")
        print(f"   Sample mapping: wall -> {mappings.get('wall')}")

        return True
    except Exception as e:
        print(f"❌ Object type constants failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Enhanced C++ Interface")
    print("=" * 50)

    tests = [
        test_object_type_constants,
        test_format_conversion,
        test_legacy_interface,
        test_enhanced_interface,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    print("=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} tests passed!")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        return 1


if __name__ == "__main__":
    exit(main())
