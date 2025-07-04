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
constexpr uint8_t CONVERTER = 4;
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

  AgentConfig create_test_agent_config() {
    AgentConfig agent_cfg;
    agent_cfg.group_name = "test_group";
    agent_cfg.group_id = 1;
    agent_cfg.freeze_duration = 100;
    agent_cfg.action_failure_penalty = 0.1f;
    agent_cfg.max_items_per_type = create_test_max_items_per_type();
    agent_cfg.resource_rewards = create_test_rewards();
    agent_cfg.resource_reward_max = create_test_resource_reward_max();
    agent_cfg.type_id = 0;
    return agent_cfg;
  }
};

// ==================== Agent Tests ====================

TEST_F(MettaGridCppTest, AgentRewards) {
  AgentConfig agent_cfg = create_test_agent_config();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));

  // Test reward values
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::ORE], 0.125f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::LASER], 0.0f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::ARMOR], 0.0f);
  EXPECT_FLOAT_EQ(agent->resource_rewards[TestItems::HEART], 1.0f);
}

TEST_F(MettaGridCppTest, AgentInventoryUpdate) {
  AgentConfig agent_cfg = create_test_agent_config();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));

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
  Grid grid(5, 10);
  EXPECT_EQ(grid.height, 5);
  EXPECT_EQ(grid.width, 10);
}

TEST_F(MettaGridCppTest, GridObjectManagement) {
  Grid grid(10, 10);

  // Create and add an agent
  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(2, 3, agent_cfg);

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
  Grid grid(10, 10);

  // Create attacker and target
  AgentConfig attacker_cfg = create_test_agent_config();
  attacker_cfg.group_name = "red";
  Agent* attacker = new Agent(2, 0, attacker_cfg);
  AgentConfig target_cfg = create_test_agent_config();
  target_cfg.group_name = "blue";
  target_cfg.group_id = 2;
  Agent* target = new Agent(0, 0, target_cfg);

  float attacker_reward = 0.0f;
  float target_reward = 0.0f;
  attacker->init(&attacker_reward);
  target->init(&target_reward);

  grid.add_object(attacker);
  grid.add_object(target);

  // Give attacker a laser
  attacker->update_inventory(TestItems::LASER, 2);
  EXPECT_EQ(attacker->inventory[TestItems::LASER], 2);

  // Give target some items and armor
  target->update_inventory(TestItems::ARMOR, 5);
  target->update_inventory(TestItems::HEART, 3);
  EXPECT_EQ(target->inventory[TestItems::ARMOR], 5);
  EXPECT_EQ(target->inventory[TestItems::HEART], 3);

  // Verify attacker orientation
  EXPECT_EQ(attacker->orientation, Orientation::Up);

  // Create attack action handler
  AttackConfig attack_cfg;
  std::map<InventoryItem, short> required_resources;
  std::map<InventoryItem, short> defense_resources;
  required_resources[TestItems::LASER] = 1;
  // In this case, defense takes 3 armor!
  defense_resources[TestItems::ARMOR] = 3;
  attack_cfg.required_resources = required_resources;
  attack_cfg.consumed_resources = required_resources;
  attack_cfg.defense_resources = defense_resources;
  Attack attack(attack_cfg);
  attack.init(&grid);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(attacker->id, 5);
  // Hitting a target with armor counts as success
  EXPECT_TRUE(success);

  // Verify that the combat material was consumed
  EXPECT_EQ(attacker->inventory[TestItems::LASER], 1);
  EXPECT_EQ(target->inventory[TestItems::ARMOR], 2);

  // Verify target was not frozen or robbed
  EXPECT_EQ(target->frozen, 0);
  EXPECT_EQ(target->inventory[TestItems::HEART], 3);

  // Attack again, now that armor is gone
  success = attack.handle_action(attacker->id, 5);
  EXPECT_TRUE(success);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory[TestItems::HEART], 0);
  EXPECT_EQ(attacker->inventory[TestItems::HEART], 3);
  // Humorously, the defender's armor was also stolen!
  EXPECT_EQ(target->inventory[TestItems::ARMOR], 0);
  EXPECT_EQ(attacker->inventory[TestItems::ARMOR], 2);
}

TEST_F(MettaGridCppTest, PutRecipeItems) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.group_name = "red";
  agent_cfg.group_id = 1;
  Agent* agent = new Agent(1, 0, agent_cfg);
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
  generator_cfg.type_id = TestItems::CONVERTER;
  generator_cfg.type_name = "generator";
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg);
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
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.group_name = "red";
  agent_cfg.group_id = 1;
  Agent* agent = new Agent(1, 0, agent_cfg);
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
  generator_cfg.type_id = TestItems::CONVERTER;
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg);
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
  Grid grid(10, 10);
  EventManager event_manager;

  // Test that event manager can be initialized
  // (This is a basic test - more complex event testing would require more setup)
  EXPECT_NO_THROW(event_manager.process_events(1));
}

// ==================== Wall/Block Tests ====================

TEST_F(MettaGridCppTest, WallCreation) {
  WallConfig wall_cfg;

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
  converter_cfg.type_id = TestItems::CONVERTER;
  converter_cfg.type_name = "converter";
  std::unique_ptr<Converter> converter(new Converter(1, 2, converter_cfg));

  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->location.r, 1);
  EXPECT_EQ(converter->location.c, 2);
  EXPECT_EQ(converter->type_id, TestItems::CONVERTER);
  EXPECT_EQ(converter->type_name, "converter");
}
