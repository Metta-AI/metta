#include <gtest/gtest.h>

#include "actions/attack.hpp"
#include "actions/get_output.hpp"
#include "actions/put_recipe_items.hpp"
#include "event.hpp"
#include "grid.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/wall.hpp"

// Test-specific inventory item type constants
namespace TestItems {
constexpr uint8_t ORE = 0;
constexpr uint8_t LASER = 1;
constexpr uint8_t ARMOR = 2;
constexpr uint8_t HEART = 3;
}  // namespace TestItems

// Pure C++ tests without any Python/pybind dependencies - we will test those with pytest
class MettaGridCppTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

  // Helper function to create test max_items_per_type map
  std::map<uint8_t, uint8_t> create_test_max_items_per_type() {
    std::map<uint8_t, uint8_t> max_items_per_type;
    max_items_per_type[TestItems::ORE] = 50;
    max_items_per_type[TestItems::LASER] = 50;
    max_items_per_type[TestItems::ARMOR] = 50;
    max_items_per_type[TestItems::HEART] = 50;
    return max_items_per_type;
  }

  // Helper function to create test rewards map
  std::map<uint8_t, float> create_test_rewards() {
    std::map<uint8_t, float> rewards;
    rewards[TestItems::ORE] = 0.125f;
    rewards[TestItems::LASER] = 0.0f;
    rewards[TestItems::ARMOR] = 0.0f;
    rewards[TestItems::HEART] = 1.0f;
    return rewards;
  }

  // Helper function to create test resource_reward_max map
  std::map<uint8_t, float> create_test_resource_reward_max() {
    std::map<uint8_t, float> resource_reward_max;
    resource_reward_max[TestItems::ORE] = 10.0f;
    resource_reward_max[TestItems::LASER] = 10.0f;
    resource_reward_max[TestItems::ARMOR] = 10.0f;
    resource_reward_max[TestItems::HEART] = 10.0f;
    return resource_reward_max;
  }

  std::vector<std::string> create_test_inventory_item_names() {
    return {"ore", "laser", "armor", "heart"};
  }
};

// ==================== Agent Tests ====================

TEST_F(MettaGridCppTest, AgentRewards) {
  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();

  std::unique_ptr<Agent> agent(new Agent(
      0, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "test_group", 1, inventory_item_names));

  // Test reward values
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::ORE], 0.125f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::LASER], 0.0f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::ARMOR], 0.0f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::HEART], 1.0f);
}

TEST_F(MettaGridCppTest, AgentInventoryUpdate) {
  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();

  std::unique_ptr<Agent> agent(new Agent(
      0, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "test_group", 1, inventory_item_names));

  float dummy_reward = 0.0f;
  agent->init(&dummy_reward);

  // Test adding items
  int delta = agent->update_inventory(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 5);

  // Test removing items
  delta = agent->update_inventory(TestItems::ORE, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 3);

  // Test hitting zero
  delta = agent->update_inventory(TestItems::ORE, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  // check that the item is not in the inventory
  EXPECT_EQ(agent->inventory.find(TestItems::ORE), agent->inventory.end());

  // Test hitting max_items_per_type limit
  agent->update_inventory(TestItems::ORE, 30);
  delta = agent->update_inventory(TestItems::ORE, 50);  // max_items_per_type is 50
  EXPECT_EQ(delta, 20);                                 // Should only add up to max_items_per_type
  EXPECT_EQ(agent->inventory[TestItems::ORE], 50);
}

// ==================== Grid Tests ====================

TEST_F(MettaGridCppTest, GridCreation) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 5, layer_for_type_id);

  EXPECT_EQ(grid.width, 10);
  EXPECT_EQ(grid.height, 5);
  EXPECT_EQ(grid.num_layers, GridLayer::GridLayerCount);
}

TEST_F(MettaGridCppTest, GridObjectManagement) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 10, layer_for_type_id);

  // Create and add an agent
  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();
  Agent* agent = new Agent(
      2, 3, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "test_group", 1, inventory_item_names);

  grid.add_object(agent);

  EXPECT_NE(agent->id, 0);  // Should have been assigned a valid ID
  EXPECT_EQ(agent->location.r, 2);
  EXPECT_EQ(agent->location.c, 3);

  // Verify we can retrieve the agent
  auto retrieved_agent = grid.object(agent->id);
  EXPECT_EQ(retrieved_agent, agent);

  // Verify it's at the expected location
  auto agent_at_location = grid.object_at(GridLocation(2, 3, GridLayer::Agent_Layer));
  EXPECT_EQ(agent_at_location, agent);
}

// ==================== Action Tests ====================

TEST_F(MettaGridCppTest, AttackAction) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 10, layer_for_type_id);

  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();

  // Create attacker and target
  Agent* attacker =
      new Agent(2, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "red", 1, inventory_item_names);
  Agent* target =
      new Agent(0, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "blue", 2, inventory_item_names);

  float attacker_reward = 0.0f;
  float target_reward = 0.0f;
  attacker->init(&attacker_reward);
  target->init(&target_reward);

  grid.add_object(attacker);
  grid.add_object(target);

  // Give attacker a laser
  attacker->update_inventory(TestItems::LASER, 1);
  EXPECT_EQ(attacker->inventory[TestItems::LASER], 1);

  // Give target some items
  target->update_inventory(TestItems::ORE, 2);
  target->update_inventory(TestItems::HEART, 3);
  EXPECT_EQ(target->inventory[TestItems::ORE], 2);
  EXPECT_EQ(target->inventory[TestItems::HEART], 3);

  // Verify attacker orientation
  EXPECT_EQ(attacker->orientation, Orientation::Up);

  // Create attack action handler
  ActionConfig attack_cfg;
  Attack attack(attack_cfg, TestItems::LASER, TestItems::ARMOR);
  attack.init(&grid);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(attacker->id, 5);
  EXPECT_TRUE(success);

  // Verify laser was consumed
  EXPECT_EQ(attacker->inventory[TestItems::LASER], 0);

  // Verify target was frozen
  EXPECT_GT(target->frozen, 0);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory[TestItems::ORE], 0);
  EXPECT_EQ(target->inventory[TestItems::HEART], 0);
  EXPECT_EQ(attacker->inventory[TestItems::ORE], 2);
  EXPECT_EQ(attacker->inventory[TestItems::HEART], 3);
}

TEST_F(MettaGridCppTest, PutRecipeItems) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 10, layer_for_type_id);

  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();

  Agent* agent =
      new Agent(1, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "red", 1, inventory_item_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  grid.add_object(agent);

  // Create a generator that takes red ore and outputs batteries
  ConverterConfig generator_cfg;
  generator_cfg.recipe_input[TestItems::ORE] = 1;
  generator_cfg.recipe_output[TestItems::ARMOR] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg.max_output = 0;
  generator_cfg.conversion_ticks = 1;
  generator_cfg.cooldown = 10;
  generator_cfg.initial_items = 0;
  generator_cfg.color = 0;
  generator_cfg.inventory_item_names = inventory_item_names;
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorRedT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(TestItems::ORE, 1);
  agent->update_inventory(TestItems::HEART, 1);

  // Create put_recipe_items action handler
  ActionConfig put_cfg;
  PutRecipeItems put(put_cfg);
  put.init(&grid);

  // Test putting matching items
  bool success = put.handle_action(agent->id, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 0);      // Ore consumed
  EXPECT_EQ(agent->inventory[TestItems::HEART], 1);    // Heart unchanged
  EXPECT_EQ(generator->inventory[TestItems::ORE], 1);  // Ore added to generator

  // Test putting non-matching items
  success = put.handle_action(agent->id, 0);
  EXPECT_FALSE(success);                                 // Should fail since we only have heart left
  EXPECT_EQ(agent->inventory[TestItems::HEART], 1);      // Heart unchanged
  EXPECT_EQ(generator->inventory[TestItems::HEART], 0);  // No heart in generator
}

TEST_F(MettaGridCppTest, GetOutput) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 10, layer_for_type_id);

  auto max_items_per_type = create_test_max_items_per_type();
  auto rewards = create_test_rewards();
  auto resource_reward_max = create_test_resource_reward_max();
  auto inventory_item_names = create_test_inventory_item_names();

  Agent* agent =
      new Agent(1, 0, 100, 0.1f, max_items_per_type, rewards, resource_reward_max, "red", 1, inventory_item_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  grid.add_object(agent);

  // Create a generator with initial output
  ConverterConfig generator_cfg;
  generator_cfg.recipe_input[TestItems::ORE] = 1;
  generator_cfg.recipe_output[TestItems::ARMOR] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg.max_output = 1;
  generator_cfg.conversion_ticks = 1;
  generator_cfg.cooldown = 10;
  generator_cfg.initial_items = 1;
  generator_cfg.color = 0;
  generator_cfg.inventory_item_names = inventory_item_names;
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorRedT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(TestItems::ORE, 1);

  // Create get_output action handler
  ActionConfig get_cfg;
  GetOutput get(get_cfg);
  get.init(&grid);

  // Test getting output
  bool success = get.handle_action(agent->id, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 1);        // Still have ore
  EXPECT_EQ(agent->inventory[TestItems::ARMOR], 1);      // Also have armor
  EXPECT_EQ(generator->inventory[TestItems::ARMOR], 0);  // Generator gave away its armor
}

// ==================== Event System Tests ====================

TEST_F(MettaGridCppTest, EventManager) {
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }

  Grid grid(10, 10, layer_for_type_id);
  EventManager event_manager;

  // Test that event manager can be initialized
  // (This is a basic test - more complex event testing would require more setup)
  EXPECT_NO_THROW(event_manager.process_events(1));
}

// ==================== Object Type Tests ====================

TEST_F(MettaGridCppTest, ObjectTypes) {
  // Test that object type constants are properly defined
  EXPECT_NE(ObjectType::AgentT, ObjectType::WallT);
  EXPECT_NE(ObjectType::AgentT, ObjectType::GeneratorRedT);
  EXPECT_NE(ObjectType::WallT, ObjectType::GeneratorRedT);

  // Test that object layers are properly mapped
  EXPECT_TRUE(ObjectLayers.find(ObjectType::AgentT) != ObjectLayers.end());
  EXPECT_TRUE(ObjectLayers.find(ObjectType::WallT) != ObjectLayers.end());
  EXPECT_TRUE(ObjectLayers.find(ObjectType::GeneratorRedT) != ObjectLayers.end());
}

// ==================== Wall/Block Tests ====================

TEST_F(MettaGridCppTest, WallCreation) {
  ObjectConfig wall_cfg;

  std::unique_ptr<Wall> wall(new Wall(2, 3, wall_cfg));

  ASSERT_NE(wall, nullptr);
  EXPECT_EQ(wall->location.r, 2);
  EXPECT_EQ(wall->location.c, 3);
}

// ==================== Converter Tests ====================

TEST_F(MettaGridCppTest, ConverterCreation) {
  ConverterConfig converter_cfg;
  converter_cfg.recipe_input[TestItems::ORE] = 2;
  converter_cfg.recipe_output[TestItems::ARMOR] = 1;
  converter_cfg.conversion_ticks = 5;
  converter_cfg.cooldown = 10;
  converter_cfg.initial_items = 0;
  converter_cfg.color = 0;
  converter_cfg.inventory_item_names = create_test_inventory_item_names();
  std::unique_ptr<Converter> converter(new Converter(1, 2, converter_cfg, ObjectType::GeneratorRedT));

  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->location.r, 1);
  EXPECT_EQ(converter->location.c, 2);
}
