#include <gtest/gtest.h>
#include <random>
#include <map>

#include "mettagrid/objects/inventory_list.hpp"
#include "mettagrid/objects/constants.hpp"
#include "mettagrid/event.hpp"

// Test fixture for InventoryList
class InventoryListTest : public ::testing::Test {
protected:
  void SetUp() override {
    inventory_list = std::make_unique<InventoryList>();
    rng = std::make_unique<std::mt19937>(42); // Fixed seed for reproducible tests

    inventory_list->set_rng(rng.get());
  }

  void TearDown() override {
    inventory_list.reset();
    rng.reset();
  }

  std::unique_ptr<InventoryList> inventory_list;
  std::unique_ptr<std::mt19937> rng;
};

// Test constructor and basic initialization
TEST_F(InventoryListTest, Constructor) {
  // Create a fresh InventoryList without SetUp dependencies
  InventoryList fresh_list;

  EXPECT_EQ(1, fresh_list.next_resource_id);
  EXPECT_TRUE(fresh_list.inventory.empty());
  EXPECT_TRUE(fresh_list.resource_instances.empty());
  EXPECT_TRUE(fresh_list.item_to_resources.empty());
  EXPECT_EQ(nullptr, fresh_list.rng);
  EXPECT_EQ(nullptr, fresh_list.event_manager);
}

// Test setting RNG and event manager
TEST_F(InventoryListTest, SetDependencies) {
  inventory_list->set_rng(rng.get());

  EXPECT_EQ(rng.get(), inventory_list->rng);
  EXPECT_EQ(nullptr, inventory_list->event_manager);
}

// Test resource instance creation
TEST_F(InventoryListTest, CreateResourceInstance) {
  InventoryItem item_type = 1;
  unsigned int timestep = 10;

  uint64_t resource_id = inventory_list->create_resource_instance(item_type, timestep);

  EXPECT_EQ(1, resource_id);
  EXPECT_EQ(2, inventory_list->next_resource_id);

  // Check resource instance was created
  auto it = inventory_list->resource_instances.find(resource_id);
  EXPECT_NE(it, inventory_list->resource_instances.end());
  EXPECT_EQ(resource_id, it->second.id);
  EXPECT_EQ(item_type, it->second.item_type);
  EXPECT_EQ(timestep, it->second.creation_timestep);

  // Check item_to_resources mapping
  auto item_it = inventory_list->item_to_resources.find(item_type);
  EXPECT_NE(item_it, inventory_list->item_to_resources.end());
  EXPECT_EQ(1, item_it->second.size());
  EXPECT_EQ(resource_id, item_it->second[0]);
}

// Test multiple resource instances
TEST_F(InventoryListTest, CreateMultipleResourceInstances) {
  InventoryItem item_type = 2;
  unsigned int timestep = 5;

  uint64_t id1 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id2 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id3 = inventory_list->create_resource_instance(item_type, timestep);

  EXPECT_EQ(1, id1);
  EXPECT_EQ(2, id2);
  EXPECT_EQ(3, id3);
  EXPECT_EQ(4, inventory_list->next_resource_id);

  // Check all instances exist
  EXPECT_EQ(3, inventory_list->resource_instances.size());
  EXPECT_EQ(3, inventory_list->item_to_resources[item_type].size());

  // Check item_to_resources contains all IDs
  const auto& resources = inventory_list->item_to_resources[item_type];
  EXPECT_TRUE(std::find(resources.begin(), resources.end(), id1) != resources.end());
  EXPECT_TRUE(std::find(resources.begin(), resources.end(), id2) != resources.end());
  EXPECT_TRUE(std::find(resources.begin(), resources.end(), id3) != resources.end());
}

// Test resource instance removal
TEST_F(InventoryListTest, RemoveResourceInstance) {
  InventoryItem item_type = 3;
  unsigned int timestep = 15;

  uint64_t id1 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id2 = inventory_list->create_resource_instance(item_type, timestep);

  // Remove first instance
  inventory_list->remove_resource_instance(id1);

  EXPECT_EQ(1, inventory_list->resource_instances.size());
  EXPECT_EQ(1, inventory_list->item_to_resources[item_type].size());
  EXPECT_EQ(id2, inventory_list->item_to_resources[item_type][0]);

  // Remove second instance
  inventory_list->remove_resource_instance(id2);

  EXPECT_TRUE(inventory_list->resource_instances.empty());
  EXPECT_TRUE(inventory_list->item_to_resources.empty());
}

// Test removing non-existent resource instance
TEST_F(InventoryListTest, RemoveNonExistentResourceInstance) {
  // Should not crash or throw
  inventory_list->remove_resource_instance(999);

  EXPECT_TRUE(inventory_list->resource_instances.empty());
  EXPECT_TRUE(inventory_list->item_to_resources.empty());
}

// Test get_random_resource_id
TEST_F(InventoryListTest, GetRandomResourceId) {
  InventoryItem item_type = 4;
  unsigned int timestep = 20;

  // No resources initially
  EXPECT_EQ(0, inventory_list->get_random_resource_id(item_type));

  // Add some resources
  uint64_t id1 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id2 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id3 = inventory_list->create_resource_instance(item_type, timestep);

  // Should return one of the valid IDs
  uint64_t random_id = inventory_list->get_random_resource_id(item_type);
  EXPECT_TRUE(random_id == id1 || random_id == id2 || random_id == id3);

  // Test with different item type
  EXPECT_EQ(0, inventory_list->get_random_resource_id(99));
}

// Test get_random_resource_id with no RNG (fallback to rand)
TEST_F(InventoryListTest, GetRandomResourceIdNoRNG) {
  inventory_list->set_rng(nullptr); // Remove RNG

  InventoryItem item_type = 5;
  unsigned int timestep = 25;

  uint64_t id1 = inventory_list->create_resource_instance(item_type, timestep);
  uint64_t id2 = inventory_list->create_resource_instance(item_type, timestep);

  // Should still work with rand() fallback
  uint64_t random_id = inventory_list->get_random_resource_id(item_type);
  EXPECT_TRUE(random_id == id1 || random_id == id2);
}

// Test create_and_schedule_resources without event manager
TEST_F(InventoryListTest, CreateAndScheduleResourcesNoEventManager) {
  inventory_list->set_event_manager(nullptr);

  InventoryItem item_type = 6;
  int count = 3;
  unsigned int timestep = 30;
  std::map<InventoryItem, float> resource_loss_prob;
  GridObjectId object_id = 123;

  inventory_list->create_and_schedule_resources(item_type, count, timestep, resource_loss_prob, object_id);

  // Should not create resource instances when event_manager is null
  EXPECT_EQ(0, inventory_list->resource_instances.size());
}

// Test create_and_schedule_resources without RNG
TEST_F(InventoryListTest, CreateAndScheduleResourcesNoRNG) {
  inventory_list->set_rng(nullptr);

  InventoryItem item_type = 7;
  int count = 2;
  unsigned int timestep = 35;
  std::map<InventoryItem, float> resource_loss_prob = {{item_type, 0.1f}};
  GridObjectId object_id = 456;

  inventory_list->create_and_schedule_resources(item_type, count, timestep, resource_loss_prob, object_id);

  // Should not create resource instances when rng is null
  EXPECT_EQ(0, inventory_list->resource_instances.size());
}

// Test create_and_schedule_resources with zero probability
TEST_F(InventoryListTest, CreateAndScheduleResourcesZeroProbability) {
  InventoryItem item_type = 8;
  int count = 2;
  unsigned int timestep = 40;
  std::map<InventoryItem, float> resource_loss_prob = {{item_type, 0.0f}};
  GridObjectId object_id = 789;

  inventory_list->create_and_schedule_resources(item_type, count, timestep, resource_loss_prob, object_id);

  // Should not create resource instances without event manager (even with zero probability)
  EXPECT_EQ(0, inventory_list->resource_instances.size());
}

// Test update_inventory with positive delta (no event manager)
TEST_F(InventoryListTest, UpdateInventoryPositiveDelta) {
  InventoryItem item_type = 10;
  InventoryDelta delta = 3;
  std::map<InventoryItem, float> resource_loss_prob = {{item_type, 0.1f}};
  GridObjectId object_id = 202;

  InventoryDelta result = inventory_list->update_inventory(item_type, delta, resource_loss_prob, object_id);

  EXPECT_EQ(3, result);
  EXPECT_EQ(3, inventory_list->inventory[item_type]);
  // Should not create resource instances without event manager
  EXPECT_EQ(0, inventory_list->resource_instances.size());
}

// Test update_inventory with negative delta
TEST_F(InventoryListTest, UpdateInventoryNegativeDelta) {
  InventoryItem item_type = 11;

  // First add some resources
  inventory_list->inventory[item_type] = 5;
  for (int i = 0; i < 5; i++) {
    inventory_list->create_resource_instance(item_type, 60);
  }

  InventoryDelta delta = -2;
  std::map<InventoryItem, float> resource_loss_prob;
  GridObjectId object_id = 303;

  InventoryDelta result = inventory_list->update_inventory(item_type, delta, resource_loss_prob, object_id);

  EXPECT_EQ(-2, result);
  EXPECT_EQ(3, inventory_list->inventory[item_type]);
  EXPECT_EQ(3, inventory_list->resource_instances.size());
}

// Test update_inventory with zero delta
TEST_F(InventoryListTest, UpdateInventoryZeroDelta) {
  InventoryItem item_type = 12;
  inventory_list->inventory[item_type] = 2;

  InventoryDelta delta = 0;
  std::map<InventoryItem, float> resource_loss_prob;
  GridObjectId object_id = 404;

  InventoryDelta result = inventory_list->update_inventory(item_type, delta, resource_loss_prob, object_id);

  EXPECT_EQ(0, result);
  EXPECT_EQ(2, inventory_list->inventory[item_type]);
}

// Test update_inventory with clamping
TEST_F(InventoryListTest, UpdateInventoryClamping) {
  InventoryItem item_type = 13;
  inventory_list->inventory[item_type] = 250;

  InventoryDelta delta = 10; // Would exceed uint8_t max
  std::map<InventoryItem, float> resource_loss_prob;
  GridObjectId object_id = 505;

  InventoryDelta result = inventory_list->update_inventory(item_type, delta, resource_loss_prob, object_id);

  // Should be clamped to max value
  EXPECT_EQ(255, inventory_list->inventory[item_type]);
  EXPECT_EQ(5, result); // Only 5 was actually added
}

// Test update_inventory with negative clamping
TEST_F(InventoryListTest, UpdateInventoryNegativeClamping) {
  InventoryItem item_type = 14;
  inventory_list->inventory[item_type] = 2;

  InventoryDelta delta = -10; // Would go negative
  std::map<InventoryItem, float> resource_loss_prob;
  GridObjectId object_id = 606;

  InventoryDelta result = inventory_list->update_inventory(item_type, delta, resource_loss_prob, object_id);

  // Should be clamped to 0 and item should be erased from inventory
  EXPECT_EQ(-2, result); // Only 2 was actually removed
  EXPECT_TRUE(inventory_list->inventory.find(item_type) == inventory_list->inventory.end());
}

// Test initialize_resource_instances without event manager
TEST_F(InventoryListTest, InitializeResourceInstancesNoEventManager) {
  inventory_list->set_event_manager(nullptr);

  InventoryItem item_type = 17;
  inventory_list->inventory[item_type] = 2;

  std::map<InventoryItem, float> resource_loss_prob = {{item_type, 0.1f}};
  GridObjectId object_id = 808;

  inventory_list->initialize_resource_instances(resource_loss_prob, object_id);

  // Should not create resource instances without event manager
  EXPECT_TRUE(inventory_list->resource_instances.empty());
}

// Test clear method
TEST_F(InventoryListTest, Clear) {
  InventoryItem item_type = 18;

  // Set up some data
  inventory_list->inventory[item_type] = 3;
  inventory_list->create_resource_instance(item_type, 80);
  inventory_list->create_resource_instance(item_type, 80);
  inventory_list->next_resource_id = 10;

  inventory_list->clear();

  EXPECT_TRUE(inventory_list->inventory.empty());
  EXPECT_TRUE(inventory_list->resource_instances.empty());
  EXPECT_TRUE(inventory_list->item_to_resources.empty());
  EXPECT_EQ(1, inventory_list->next_resource_id);
}

// Test resource instance ID uniqueness across different item types
TEST_F(InventoryListTest, ResourceInstanceIdUniqueness) {
  InventoryItem item_type1 = 19;
  InventoryItem item_type2 = 20;
  unsigned int timestep = 90;

  uint64_t id1 = inventory_list->create_resource_instance(item_type1, timestep);
  uint64_t id2 = inventory_list->create_resource_instance(item_type2, timestep);
  uint64_t id3 = inventory_list->create_resource_instance(item_type1, timestep);

  EXPECT_NE(id1, id2);
  EXPECT_NE(id1, id3);
  EXPECT_NE(id2, id3);
  EXPECT_EQ(4, inventory_list->next_resource_id);
}

// Test edge case: removing resource instance that doesn't exist in item_to_resources
TEST_F(InventoryListTest, RemoveResourceInstanceInconsistentState) {
  InventoryItem item_type = 21;
  unsigned int timestep = 100;

  // Manually create inconsistent state
  uint64_t resource_id = 1;
  inventory_list->resource_instances[resource_id] = {resource_id, item_type, timestep};
  // Don't add to item_to_resources

  // Should not crash
  inventory_list->remove_resource_instance(resource_id);

  EXPECT_TRUE(inventory_list->resource_instances.empty());
  EXPECT_TRUE(inventory_list->item_to_resources.empty());
}

// Test edge case: empty item_to_resources vector after removal
TEST_F(InventoryListTest, EmptyItemToResourcesAfterRemoval) {
  InventoryItem item_type = 22;
  unsigned int timestep = 110;

  uint64_t resource_id = inventory_list->create_resource_instance(item_type, timestep);

  // Verify it exists
  EXPECT_EQ(1, inventory_list->item_to_resources[item_type].size());

  // Remove it
  inventory_list->remove_resource_instance(resource_id);

  // Should remove the entire item_type entry
  EXPECT_TRUE(inventory_list->item_to_resources.empty());
}
