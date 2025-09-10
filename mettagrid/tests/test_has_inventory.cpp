#include <gtest/gtest.h>
#include <map>

#include "mettagrid/objects/box.hpp"
#include "mettagrid/objects/converter.hpp"
#include "mettagrid/objects/converter_config.hpp"

// Test fixture for HasInventory classes
class HasInventoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up test data
  }

  void TearDown() override {
    // Cleanup if needed
  }
};

// Test Box class with HasInventory interface
TEST_F(HasInventoryTest, BoxHasInventoryInterface) {
  // Create a BoxConfig
  std::map<InventoryItem, InventoryQuantity> returned_resources = {{1, 5}, {2, 3}};
  BoxConfig config(1, "test_box", returned_resources);

  // Create a Box
  Box box(10, 20, config, 123, 1);

  // Test that Box implements HasInventory interface
  EXPECT_TRUE(box.inventory_is_accessible());

  // Test that we can access the inventory through the interface
  auto& inventory = box.inventory;
  EXPECT_TRUE(inventory.empty()); // Box starts with empty inventory

  // Test that we can access the InventoryList
  auto& inventory_list = box.get_inventory_list();
  EXPECT_TRUE(inventory_list.inventory.empty());
  EXPECT_TRUE(inventory_list.resource_instances.empty());

  // Test inventory update
  InventoryDelta delta = box.update_inventory(1, 3);
  EXPECT_EQ(3, delta);
  EXPECT_EQ(3, inventory[1]);

  // Test that the inventory is accessible through both interfaces
  EXPECT_EQ(3, box.inventory[1]);
  EXPECT_EQ(3, box.get_inventory_list().inventory[1]);
}

// Test that Box can be used polymorphically through HasInventory
TEST_F(HasInventoryTest, PolymorphicUsage) {
  // Create a Box
  std::map<InventoryItem, InventoryQuantity> returned_resources = {{1, 5}};
  BoxConfig box_config(1, "test_box", returned_resources);
  Box box(10, 20, box_config, 123, 1);

  // Test polymorphic usage
  HasInventory* inventory_obj = &box;

  // All objects should be accessible
  EXPECT_TRUE(inventory_obj->inventory_is_accessible());

    // All objects should have empty inventory initially
    EXPECT_TRUE(inventory_obj->get_inventory_list().inventory.empty());

    // All objects should support inventory updates
    InventoryDelta delta = inventory_obj->update_inventory(1, 3);
    EXPECT_EQ(3, delta);
    EXPECT_EQ(3, inventory_obj->get_inventory_list().inventory[1]);

  // All objects should provide access to InventoryList
  auto& inventory_list = inventory_obj->get_inventory_list();
  EXPECT_EQ(3, inventory_list.inventory[1]);
}
