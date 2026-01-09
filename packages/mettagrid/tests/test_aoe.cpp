#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/aoe_config.hpp"
#include "core/aoe_helper.hpp"
#include "core/grid_object.hpp"
#include "objects/alignable.hpp"
#include "objects/collective.hpp"
#include "objects/collective_config.hpp"
#include "objects/has_inventory.hpp"
#include "objects/inventory_config.hpp"

using namespace mettagrid;

// Resource names for testing
static std::vector<std::string> test_resource_names = {"health", "energy", "gold"};

// Simple GridObject subclass that has an inventory and is alignable
class TestAOEObject : public GridObject, public HasInventory, public Alignable {
public:
  explicit TestAOEObject(const std::string& type = "test_object", GridCoord row = 0, GridCoord col = 0)
      : HasInventory(create_inventory_config()) {
    type_name = type;
    location.r = row;
    location.c = col;
  }

  static InventoryConfig create_inventory_config() {
    InventoryConfig config;
    config.limit_defs.push_back(LimitDef({0}, 1000));  // health
    config.limit_defs.push_back(LimitDef({1}, 1000));  // energy
    config.limit_defs.push_back(LimitDef({2}, 1000));  // gold
    return config;
  }
};

// Helper to create a collective config
CollectiveConfig create_test_collective_config(const std::string& name) {
  CollectiveConfig config;
  config.name = name;
  config.inventory_config.limit_defs.push_back(LimitDef({0}, 1000));
  config.inventory_config.limit_defs.push_back(LimitDef({1}, 1000));
  config.inventory_config.limit_defs.push_back(LimitDef({2}, 1000));
  return config;
}

void test_aoe_effect_grid_creation() {
  std::cout << "Testing AOEEffectGrid creation..." << std::endl;

  AOEEffectGrid grid(10, 10);

  // No effects should be registered initially
  GridLocation loc(5, 5);
  assert(grid.effect_count_at(loc) == 0);

  std::cout << "✓ AOEEffectGrid creation test passed" << std::endl;
}

void test_register_source_basic() {
  std::cout << "Testing register_source basic..." << std::endl;

  AOEEffectGrid grid(10, 10);

  // Create a source object at position (5, 5)
  TestAOEObject source("healer", 5, 5);

  // Create a config with range 1
  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));  // +10 health

  grid.register_source(source, config);

  // Effect should be registered at source location and all cells within L-infinity distance 1
  // Cells affected: all 9 cells in the 3x3 square centered at (5,5)
  assert(grid.effect_count_at(GridLocation(5, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(4, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 6)) == 1);

  // Diagonal cells ARE affected with L-infinity distance
  assert(grid.effect_count_at(GridLocation(4, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(4, 6)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 6)) == 1);

  // Cells at distance 2 should NOT be affected
  assert(grid.effect_count_at(GridLocation(3, 5)) == 0);
  assert(grid.effect_count_at(GridLocation(7, 5)) == 0);

  std::cout << "✓ register_source basic test passed" << std::endl;
}

void test_register_source_range_2() {
  std::cout << "Testing register_source with range 2..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);

  AOEConfig config;
  config.radius = 2;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source, config);

  // With L-infinity distance, range 2 creates a 5x5 square centered on source
  // Center
  assert(grid.effect_count_at(GridLocation(5, 5)) == 1);

  // Distance 1 (cardinal)
  assert(grid.effect_count_at(GridLocation(4, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 6)) == 1);

  // Distance 2 (cardinal)
  assert(grid.effect_count_at(GridLocation(3, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(7, 5)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 3)) == 1);
  assert(grid.effect_count_at(GridLocation(5, 7)) == 1);

  // Distance 1 (diagonal) - L-infinity distance is 1
  assert(grid.effect_count_at(GridLocation(4, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(4, 6)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 4)) == 1);
  assert(grid.effect_count_at(GridLocation(6, 6)) == 1);

  // Distance 2 (diagonal) - L-infinity distance is 2
  assert(grid.effect_count_at(GridLocation(3, 3)) == 1);
  assert(grid.effect_count_at(GridLocation(3, 7)) == 1);
  assert(grid.effect_count_at(GridLocation(7, 3)) == 1);
  assert(grid.effect_count_at(GridLocation(7, 7)) == 1);

  // Distance 3 (should NOT be affected)
  assert(grid.effect_count_at(GridLocation(2, 5)) == 0);
  assert(grid.effect_count_at(GridLocation(5, 2)) == 0);
  assert(grid.effect_count_at(GridLocation(2, 2)) == 0);  // Corner at distance 3

  std::cout << "✓ register_source range 2 test passed" << std::endl;
}

void test_unregister_source() {
  std::cout << "Testing unregister_source..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source, config);
  assert(grid.effect_count_at(GridLocation(5, 5)) == 1);

  grid.unregister_source(source);
  assert(grid.effect_count_at(GridLocation(5, 5)) == 0);
  assert(grid.effect_count_at(GridLocation(4, 5)) == 0);
  assert(grid.effect_count_at(GridLocation(6, 5)) == 0);

  std::cout << "✓ unregister_source test passed" << std::endl;
}

void test_multiple_sources() {
  std::cout << "Testing multiple sources..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source1("healer1", 5, 5);
  TestAOEObject source2("healer2", 6, 5);  // Adjacent to source1

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source1, config);
  grid.register_source(source2, config);

  // Source1's location is affected by both sources
  assert(grid.effect_count_at(GridLocation(5, 5)) == 2);

  // Source2's location is affected by both sources
  assert(grid.effect_count_at(GridLocation(6, 5)) == 2);

  // (4, 5) only affected by source1
  assert(grid.effect_count_at(GridLocation(4, 5)) == 1);

  // (7, 5) only affected by source2
  assert(grid.effect_count_at(GridLocation(7, 5)) == 1);

  std::cout << "✓ multiple sources test passed" << std::endl;
}

void test_apply_effects_basic() {
  std::cout << "Testing apply_effects_at basic..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);
  TestAOEObject target("agent", 5, 6);  // Adjacent to source

  // Set initial health
  target.inventory.update(0, 100);
  assert(target.inventory.amount(0) == 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));  // +10 health

  grid.register_source(source, config);
  grid.apply_effects_at(target.location, target);

  // Target should have gained 10 health
  assert(target.inventory.amount(0) == 110);

  std::cout << "✓ apply_effects_at basic test passed" << std::endl;
}

void test_source_does_not_affect_itself() {
  std::cout << "Testing source does not affect itself..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);
  source.inventory.update(0, 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source, config);
  grid.apply_effects_at(source.location, source);

  // Source should NOT have gained health (it's the source of the effect)
  assert(source.inventory.amount(0) == 100);

  std::cout << "✓ source does not affect itself test passed" << std::endl;
}

void test_multiple_deltas() {
  std::cout << "Testing multiple resource deltas..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("buff_station", 5, 5);
  TestAOEObject target("agent", 5, 6);

  target.inventory.update(0, 100);  // health
  target.inventory.update(1, 50);   // energy

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));  // +10 health
  config.deltas.push_back(AOEResourceDelta(1, -5));  // -5 energy

  grid.register_source(source, config);
  grid.apply_effects_at(target.location, target);

  assert(target.inventory.amount(0) == 110);  // 100 + 10
  assert(target.inventory.amount(1) == 45);   // 50 - 5

  std::cout << "✓ multiple resource deltas test passed" << std::endl;
}

void test_tag_filter() {
  std::cout << "Testing tag filter..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);
  TestAOEObject target_with_tag("agent", 5, 6);
  TestAOEObject target_without_tag("enemy", 6, 5);

  // Add tag to one target
  target_with_tag.tag_ids.push_back(42);

  target_with_tag.inventory.update(0, 100);
  target_without_tag.inventory.update(0, 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));
  config.target_tag_ids.push_back(42);  // Only affect objects with tag 42

  grid.register_source(source, config);
  grid.apply_effects_at(target_with_tag.location, target_with_tag);
  grid.apply_effects_at(target_without_tag.location, target_without_tag);

  assert(target_with_tag.inventory.amount(0) == 110);     // Affected
  assert(target_without_tag.inventory.amount(0) == 100);  // Not affected

  std::cout << "✓ tag filter test passed" << std::endl;
}

void test_alignment_filter_same_collective() {
  std::cout << "Testing alignment filter (same_collective)..." << std::endl;

  AOEEffectGrid grid(10, 10);

  CollectiveConfig coll_config = create_test_collective_config("team_a");
  Collective collective(coll_config, &test_resource_names);

  TestAOEObject source("healer", 5, 5);
  TestAOEObject target_same("agent", 5, 6);
  TestAOEObject target_different("enemy", 6, 5);

  // Both source and target_same are in the same collective
  source.setCollective(&collective);
  target_same.setCollective(&collective);

  target_same.inventory.update(0, 100);
  target_different.inventory.update(0, 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));
  config.alignment_filter = AOEAlignmentFilter::same_collective;

  grid.register_source(source, config);
  grid.apply_effects_at(target_same.location, target_same);
  grid.apply_effects_at(target_different.location, target_different);

  assert(target_same.inventory.amount(0) == 110);       // Same collective, affected
  assert(target_different.inventory.amount(0) == 100);  // No collective, not affected

  std::cout << "✓ alignment filter (same_collective) test passed" << std::endl;
}

void test_alignment_filter_different_collective() {
  std::cout << "Testing alignment filter (different_collective)..." << std::endl;

  AOEEffectGrid grid(10, 10);

  CollectiveConfig coll_config_a = create_test_collective_config("team_a");
  CollectiveConfig coll_config_b = create_test_collective_config("team_b");
  Collective collective_a(coll_config_a, &test_resource_names);
  Collective collective_b(coll_config_b, &test_resource_names);

  TestAOEObject source("damager", 5, 5);
  TestAOEObject target_same("ally", 5, 6);
  TestAOEObject target_different("enemy", 6, 5);
  TestAOEObject target_no_collective("neutral", 4, 5);

  source.setCollective(&collective_a);
  target_same.setCollective(&collective_a);
  target_different.setCollective(&collective_b);

  target_same.inventory.update(0, 100);
  target_different.inventory.update(0, 100);
  target_no_collective.inventory.update(0, 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, -10));  // -10 health (damage)
  config.alignment_filter = AOEAlignmentFilter::different_collective;

  grid.register_source(source, config);
  grid.apply_effects_at(target_same.location, target_same);
  grid.apply_effects_at(target_different.location, target_different);
  grid.apply_effects_at(target_no_collective.location, target_no_collective);

  assert(target_same.inventory.amount(0) == 100);           // Same collective, not affected
  assert(target_different.inventory.amount(0) == 90);       // Different collective, affected
  assert(target_no_collective.inventory.amount(0) == 100);  // No collective, not affected

  std::cout << "✓ alignment filter (different_collective) test passed" << std::endl;
}

void test_boundary_effects() {
  std::cout << "Testing boundary effects..." << std::endl;

  AOEEffectGrid grid(10, 10);

  // Source at corner (0, 0)
  TestAOEObject source("healer", 0, 0);

  AOEConfig config;
  config.radius = 2;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source, config);

  // Should be registered at valid cells only
  assert(grid.effect_count_at(GridLocation(0, 0)) == 1);
  assert(grid.effect_count_at(GridLocation(0, 1)) == 1);
  assert(grid.effect_count_at(GridLocation(1, 0)) == 1);
  assert(grid.effect_count_at(GridLocation(0, 2)) == 1);
  assert(grid.effect_count_at(GridLocation(2, 0)) == 1);
  assert(grid.effect_count_at(GridLocation(1, 1)) == 1);

  // Should not wrap around or go negative
  // These would be invalid locations anyway

  std::cout << "✓ boundary effects test passed" << std::endl;
}

void test_out_of_range_target() {
  std::cout << "Testing out of range target..." << std::endl;

  AOEEffectGrid grid(10, 10);

  TestAOEObject source("healer", 5, 5);
  TestAOEObject target("agent", 0, 0);  // Far from source

  target.inventory.update(0, 100);

  AOEConfig config;
  config.radius = 1;
  config.deltas.push_back(AOEResourceDelta(0, 10));

  grid.register_source(source, config);
  grid.apply_effects_at(target.location, target);

  // Target should NOT be affected (out of range)
  assert(target.inventory.amount(0) == 100);

  std::cout << "✓ out of range target test passed" << std::endl;
}

int main() {
  std::cout << "Running AOE System tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  test_aoe_effect_grid_creation();
  test_register_source_basic();
  test_register_source_range_2();
  test_unregister_source();
  test_multiple_sources();
  test_apply_effects_basic();
  test_source_does_not_affect_itself();
  test_multiple_deltas();
  test_tag_filter();
  test_alignment_filter_same_collective();
  test_alignment_filter_different_collective();
  test_boundary_effects();
  test_out_of_range_target();

  std::cout << "================================================" << std::endl;
  std::cout << "All AOE System tests passed! ✓" << std::endl;

  return 0;
}
