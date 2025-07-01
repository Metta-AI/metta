#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "grid.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"
#include "action_handler.hpp"
#include "actions/attack.hpp"

// Test fixture for memory safety tests
class MemorySafetyTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up test environment
  }

  void TearDown() override {
    // Clean up after each test
  }

  // Helper function to create test max_items_per_type map
  std::map<uint8_t, uint8_t> create_test_max_items_per_type() {
    std::map<uint8_t, uint8_t> max_items_per_type;
    max_items_per_type[0] = 50;  // ore
    max_items_per_type[1] = 50;  // laser
    max_items_per_type[2] = 50;  // armor
    max_items_per_type[3] = 50;  // heart
    return max_items_per_type;
  }

  std::map<uint8_t, float> create_test_rewards() {
    std::map<uint8_t, float> rewards;
    rewards[0] = 0.125f;  // ore
    rewards[1] = 0.0f;    // laser
    rewards[2] = 0.0f;    // armor
    rewards[3] = 1.0f;    // heart
    return rewards;
  }

  std::vector<std::string> create_test_inventory_item_names() {
    return {"ore", "laser", "armor", "heart"};
  }
};

// Test that grid properly handles out-of-bounds access
TEST_F(MemorySafetyTest, GridBoundsChecking) {
  Grid grid(10, 10);
  
  // Test out-of-bounds object access
  EXPECT_EQ(grid.object_at(GridLocation(15, 5, 0)), nullptr);
  EXPECT_EQ(grid.object_at(GridLocation(5, 15, 0)), nullptr);
  EXPECT_EQ(grid.object_at(GridLocation(5, 5, 99)), nullptr);
  
  // Test invalid object ID access
  EXPECT_EQ(grid.object(9999), nullptr);
}

// Test that add_object properly handles invalid locations
TEST_F(MemorySafetyTest, GridAddObjectBounds) {
  Grid grid(5, 5);
  
  auto max_items = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto item_names = create_test_inventory_item_names();
  
  // Try to add agent outside bounds
  Agent* agent_oob = new Agent(10, 10, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  EXPECT_FALSE(grid.add_object(agent_oob));
  
  // Since add_object failed, we need to clean up the agent ourselves
  delete agent_oob;
  
  // Try to add agent at valid location
  Agent* agent_valid = new Agent(2, 2, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  EXPECT_TRUE(grid.add_object(agent_valid));
  
  // Try to add another agent at the same location (should fail)
  Agent* agent_conflict = new Agent(2, 2, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  EXPECT_FALSE(grid.add_object(agent_conflict));
  
  // Clean up the conflicting agent
  delete agent_conflict;
}

// Test remove_object bounds checking
TEST_F(MemorySafetyTest, GridRemoveObjectSafety) {
  Grid grid(5, 5);
  
  auto max_items = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto item_names = create_test_inventory_item_names();
  
  Agent* agent = new Agent(2, 2, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  ASSERT_TRUE(grid.add_object(agent));
  
  // Test removing valid object
  auto removed = grid.remove_object(agent);
  EXPECT_NE(removed, nullptr);
  
  // Test removing null object
  auto null_removed = grid.remove_object(nullptr);
  EXPECT_EQ(null_removed, nullptr);
  
  // Agent is now owned by removed unique_ptr and will be cleaned up automatically
}

// Test dynamic_cast safety in action handlers
TEST_F(MemorySafetyTest, ActionHandlerTypeSafety) {
  Grid grid(10, 10);
  
  auto max_items = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto item_names = create_test_inventory_item_names();
  
  // Add an agent
  Agent* agent = new Agent(2, 2, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  ASSERT_TRUE(grid.add_object(agent));
  
  // Add a wall (not an agent)
  WallConfig wall_cfg;
  wall_cfg.type_id = 99;
  wall_cfg.type_name = "wall";
  wall_cfg.swappable = false;
  Wall* wall = new Wall(3, 3, wall_cfg);
  ASSERT_TRUE(grid.add_object(wall));
  
  // Create attack action handler
  ActionConfig attack_cfg;
  Attack attack(attack_cfg, 1, 2);  // laser_item_id=1, armor_item_id=2
  attack.init(&grid);
  
  // Test that action handler correctly handles agent
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  agent->update_inventory(1, 1);  // Give agent a laser
  
  // This should work (agent is valid)
  EXPECT_TRUE(attack.handle_action(agent->id, 1));
  
  // Test that action handler safely handles non-agent object
  // This should fail gracefully (wall is not an agent)
  EXPECT_FALSE(attack.handle_action(wall->id, 1));
  
  // Test invalid object ID
  EXPECT_FALSE(attack.handle_action(9999, 1));
}

// Test that unique_ptr ownership is handled correctly
TEST_F(MemorySafetyTest, UniquePointerOwnership) {
  Grid grid(5, 5);
  
  auto max_items = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto item_names = create_test_inventory_item_names();
  
  // Create agent with unique_ptr
  auto agent = std::make_unique<Agent>(2, 2, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
  Agent* agent_ptr = agent.get();
  
  // Transfer ownership to grid
  EXPECT_TRUE(grid.add_object(agent.release()));
  
  // Verify agent is accessible through grid
  EXPECT_EQ(grid.object(agent_ptr->id), agent_ptr);
  
  // Remove and verify proper ownership transfer
  auto removed = grid.remove_object(agent_ptr);
  EXPECT_NE(removed, nullptr);
  EXPECT_EQ(removed.get(), agent_ptr);
  
  // removed unique_ptr will clean up the agent automatically
}

// Test memory safety during object destruction
TEST_F(MemorySafetyTest, ObjectDestructionSafety) {
  {
    Grid grid(5, 5);
    
    auto max_items = create_test_max_items_per_type();
    auto rewards = create_test_rewards();
    auto item_names = create_test_inventory_item_names();
    
    // Add multiple objects
    for (int i = 0; i < 3; ++i) {
      Agent* agent = new Agent(i, i, 100, 0.1f, max_items, rewards, rewards, "test", 1, item_names, 0);
      EXPECT_TRUE(grid.add_object(agent));
    }
    
    // Grid destructor should clean up all objects properly
  }
  // All objects should be destroyed when grid goes out of scope
  
  // If we reach here without crashes, the test passes
  SUCCEED();
}