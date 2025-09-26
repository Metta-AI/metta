#include <gtest/gtest.h>

#include "objects/inventory.hpp"
#include "systems/stats_tracker.hpp"

class InventoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up test data
    test_inventory_[1] = 10;
    test_inventory_[2] = 5;
    test_inventory_[3] = 0;  // Should be ignored

    resource_limits_[1] = 20;
    resource_limits_[2] = 10;
    // No limit for item 3

    soul_bound_resources_ = {1};  // Item 1 is soul-bound
  }

  std::map<InventoryItem, InventoryQuantity> test_inventory_;
  std::map<InventoryItem, InventoryQuantity> resource_limits_;
  std::vector<InventoryItem> soul_bound_resources_;
};

TEST_F(InventoryTest, ConstructorEmpty) {
  Inventory inv;
  EXPECT_TRUE(inv.is_empty());
  EXPECT_EQ(inv.size(), 0);
}

TEST_F(InventoryTest, ConstructorWithInitialInventory) {
  Inventory inv(test_inventory_);
  EXPECT_FALSE(inv.is_empty());
  EXPECT_EQ(inv.size(), 2);  // Only non-zero items
  EXPECT_EQ(inv.get_quantity(1), 10);
  EXPECT_EQ(inv.get_quantity(2), 5);
  EXPECT_EQ(inv.get_quantity(3), 0);
}

TEST_F(InventoryTest, ConstructorWithLimits) {
  Inventory inv(test_inventory_, resource_limits_);
  EXPECT_EQ(inv.get_quantity(1), 10);
  EXPECT_EQ(inv.get_quantity(2), 5);
  EXPECT_TRUE(inv.has_resource_limit(1));
  EXPECT_TRUE(inv.has_resource_limit(2));
  EXPECT_FALSE(inv.has_resource_limit(3));
  EXPECT_EQ(inv.get_resource_limit(1), 20);
  EXPECT_EQ(inv.get_resource_limit(2), 10);
}

TEST_F(InventoryTest, UpdateInventoryAdd) {
  Inventory inv;
  InventoryDelta delta = inv.update_inventory(1, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(inv.get_quantity(1), 5);
  EXPECT_TRUE(inv.has_item(1));
}

TEST_F(InventoryTest, UpdateInventoryRemove) {
  Inventory inv(test_inventory_);
  InventoryDelta delta = inv.update_inventory(1, -3);
  EXPECT_EQ(delta, -3);
  EXPECT_EQ(inv.get_quantity(1), 7);

  // Remove all remaining
  delta = inv.update_inventory(1, -7);
  EXPECT_EQ(delta, -7);
  EXPECT_EQ(inv.get_quantity(1), 0);
  EXPECT_FALSE(inv.has_item(1));
}

TEST_F(InventoryTest, UpdateInventoryWithLimits) {
  Inventory inv(test_inventory_, resource_limits_);

  // Try to add more than limit allows
  InventoryDelta delta = inv.update_inventory(1, 15);  // Current: 10, Limit: 20, Trying: 15
  EXPECT_EQ(delta, 10);                                // Should only add 10 (up to limit)
  EXPECT_EQ(inv.get_quantity(1), 20);

  // Try to add when already at limit
  delta = inv.update_inventory(1, 5);
  EXPECT_EQ(delta, 0);  // Should add nothing
  EXPECT_EQ(inv.get_quantity(1), 20);
}

TEST_F(InventoryTest, UpdateInventoryNoLimits) {
  Inventory inv(test_inventory_);

  // Item 3 has no limits
  InventoryDelta delta = inv.update_inventory(3, 100);
  EXPECT_EQ(delta, 100);
  EXPECT_EQ(inv.get_quantity(3), 100);
}

TEST_F(InventoryTest, SoulBoundResources) {
  Inventory inv(test_inventory_);
  inv.set_soul_bound_resources(soul_bound_resources_);

  EXPECT_TRUE(inv.is_soul_bound(1));
  EXPECT_FALSE(inv.is_soul_bound(2));

  // Try to remove soul-bound resource
  InventoryDelta delta = inv.update_inventory(1, -5);
  EXPECT_EQ(delta, 0);                 // Should not remove anything
  EXPECT_EQ(inv.get_quantity(1), 10);  // Should remain unchanged

  // Can still add to soul-bound resources
  delta = inv.update_inventory(1, 3);
  EXPECT_EQ(delta, 3);
  EXPECT_EQ(inv.get_quantity(1), 13);
}

TEST_F(InventoryTest, ResourceLimits) {
  Inventory inv;
  inv.set_resource_limits(resource_limits_);

  EXPECT_TRUE(inv.has_resource_limit(1));
  EXPECT_EQ(inv.get_resource_limit(1), 20);
  EXPECT_FALSE(inv.has_resource_limit(3));

  // Test would_exceed_limit
  inv.update_inventory(1, 15);
  EXPECT_FALSE(inv.would_exceed_limit(1, 5));  // 15 + 5 = 20, at limit
  EXPECT_TRUE(inv.would_exceed_limit(1, 6));   // 15 + 6 = 21, exceeds limit

  // Test get_max_can_add
  EXPECT_EQ(inv.get_max_can_add(1), 5);  // 20 - 15 = 5
  inv.update_inventory(1, 5);
  EXPECT_EQ(inv.get_max_can_add(1), 0);  // At limit
}

TEST_F(InventoryTest, ValidationMethods) {
  Inventory inv(test_inventory_, resource_limits_);
  inv.set_soul_bound_resources(soul_bound_resources_);

  // Test is_valid_delta
  EXPECT_TRUE(inv.is_valid_delta(2, 5));    // Valid addition
  EXPECT_TRUE(inv.is_valid_delta(2, -3));   // Valid removal
  EXPECT_FALSE(inv.is_valid_delta(1, -3));  // Invalid: soul-bound removal
  EXPECT_FALSE(inv.is_valid_delta(1, 15));  // Invalid: would exceed limit

  // Test clamp_delta_to_limits
  EXPECT_EQ(inv.clamp_delta_to_limits(1, 15), 10);  // Should clamp to 10
  EXPECT_EQ(inv.clamp_delta_to_limits(1, 5), 5);    // Should not clamp
  EXPECT_EQ(inv.clamp_delta_to_limits(2, -3), -3);  // No limits on removal
}

TEST_F(InventoryTest, UtilityMethods) {
  Inventory inv(test_inventory_);

  EXPECT_FALSE(inv.is_empty());
  EXPECT_EQ(inv.size(), 2);
  EXPECT_EQ(inv.get_total_items(), 15);  // 10 + 5

  EXPECT_TRUE(inv.has_enough(1, 5));
  EXPECT_FALSE(inv.has_enough(1, 15));
  EXPECT_FALSE(inv.has_enough(3, 1));

  inv.clear();
  EXPECT_TRUE(inv.is_empty());
  EXPECT_EQ(inv.size(), 0);
  EXPECT_EQ(inv.get_total_items(), 0);
}

TEST_F(InventoryTest, SetInventory) {
  Inventory inv;
  inv.update_inventory(1, 5);
  inv.update_inventory(2, 3);

  std::map<InventoryItem, InventoryQuantity> new_inventory;
  new_inventory[1] = 8;
  new_inventory[3] = 2;

  inv.set_inventory(new_inventory);

  EXPECT_EQ(inv.get_quantity(1), 8);
  EXPECT_EQ(inv.get_quantity(2), 0);  // Should be removed
  EXPECT_EQ(inv.get_quantity(3), 2);
  EXPECT_EQ(inv.size(), 2);
}

TEST_F(InventoryTest, IteratorSupport) {
  Inventory inv(test_inventory_);

  // Test range-based loop
  int count = 0;
  InventoryQuantity total = 0;
  for (const auto& [item, quantity] : inv) {
    count++;
    total += quantity;
  }

  EXPECT_EQ(count, 2);
  EXPECT_EQ(total, 15);
}

TEST_F(InventoryTest, StatsTracking) {
  Inventory inv(test_inventory_);
  StatsTracker stats;
  inv.set_stats_tracker(&stats);
  inv.enable_stats_tracking(true);

  // Add some items
  inv.update_inventory(1, 3);
  inv.update_inventory(2, -2);

  // Check that stats were updated
  auto stats_dict = stats.to_dict();
  // Note: The actual stat names depend on StatsTracker implementation
  // This test assumes the resource_name method returns appropriate names
}

TEST_F(InventoryTest, CopyAndMove) {
  Inventory inv1(test_inventory_, resource_limits_);
  inv1.set_soul_bound_resources(soul_bound_resources_);

  // Test copy constructor
  Inventory inv2(inv1);
  EXPECT_EQ(inv2.get_quantity(1), 10);
  EXPECT_EQ(inv2.get_resource_limit(1), 20);
  EXPECT_TRUE(inv2.is_soul_bound(1));

  // Test assignment operator
  Inventory inv3;
  inv3 = inv1;
  EXPECT_EQ(inv3.get_quantity(1), 10);
  EXPECT_EQ(inv3.get_resource_limit(1), 20);
  EXPECT_TRUE(inv3.is_soul_bound(1));

  // Test move constructor
  Inventory inv4(std::move(inv1));
  EXPECT_EQ(inv4.get_quantity(1), 10);
  EXPECT_TRUE(inv1.is_empty());  // Moved from should be empty

  // Test move assignment
  Inventory inv5;
  inv5 = std::move(inv2);
  EXPECT_EQ(inv5.get_quantity(1), 10);
  EXPECT_TRUE(inv2.is_empty());  // Moved from should be empty
}

TEST_F(InventoryTest, EdgeCases) {
  Inventory inv;

  // Test with maximum values
  InventoryDelta large_delta = std::numeric_limits<InventoryDelta>::max();
  InventoryDelta delta = inv.update_inventory(1, large_delta);
  EXPECT_EQ(delta, large_delta);

  // Test with minimum values
  inv.update_inventory(1, 100);
  InventoryDelta small_delta = std::numeric_limits<InventoryDelta>::min();
  delta = inv.update_inventory(1, small_delta);
  EXPECT_EQ(delta, small_delta);
  EXPECT_EQ(inv.get_quantity(1), 0);  // Should be clamped to 0
}

TEST_F(InventoryTest, SoulBoundManagement) {
  Inventory inv;

  // Test adding soul-bound resources
  inv.add_soul_bound_resource(1);
  inv.add_soul_bound_resource(2);
  EXPECT_TRUE(inv.is_soul_bound(1));
  EXPECT_TRUE(inv.is_soul_bound(2));

  // Test removing soul-bound resources
  inv.remove_soul_bound_resource(1);
  EXPECT_FALSE(inv.is_soul_bound(1));
  EXPECT_TRUE(inv.is_soul_bound(2));

  // Test setting soul-bound resources
  std::vector<InventoryItem> new_soul_bound = {3, 4};
  inv.set_soul_bound_resources(new_soul_bound);
  EXPECT_FALSE(inv.is_soul_bound(1));
  EXPECT_FALSE(inv.is_soul_bound(2));
  EXPECT_TRUE(inv.is_soul_bound(3));
  EXPECT_TRUE(inv.is_soul_bound(4));
}
