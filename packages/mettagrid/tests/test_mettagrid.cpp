#include <gtest/gtest.h>

#include <random>

#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "actions/get_output.hpp"
#include "actions/noop.hpp"
#include "actions/put_recipe_items.hpp"
#include "config/mettagrid_config.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/types.hpp"
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

namespace TestRewards {
constexpr float ORE = 0.125f;
constexpr float LASER = 0.0f;
constexpr float ARMOR = 0.0f;
constexpr float HEART = 1.0f;
}  // namespace TestRewards

// Pure C++ tests without any Python/pybind dependencies - we will test those with pytest
class MettaGridCppTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

  // Helper function to create test resource_limits map
  std::map<uint8_t, InventoryQuantity> create_test_resource_limits() {
    std::map<uint8_t, InventoryQuantity> resource_limits;
    resource_limits[TestItems::ORE] = 50;
    resource_limits[TestItems::LASER] = 50;
    resource_limits[TestItems::ARMOR] = 50;
    resource_limits[TestItems::HEART] = 50;
    return resource_limits;
  }

  // Helper function to create test rewards map
  std::map<uint8_t, RewardType> create_test_rewards() {
    std::map<uint8_t, RewardType> rewards;
    rewards[TestItems::ORE] = TestRewards::ORE;
    rewards[TestItems::LASER] = TestRewards::LASER;
    rewards[TestItems::ARMOR] = TestRewards::ARMOR;
    rewards[TestItems::HEART] = TestRewards::HEART;
    return rewards;
  }

  // Helper function to create test resource_reward_max map
  std::map<uint8_t, RewardType> create_test_resource_reward_max() {
    std::map<uint8_t, RewardType> resource_reward_max;
    resource_reward_max[TestItems::ORE] = 10.0f;
    resource_reward_max[TestItems::LASER] = 10.0f;
    resource_reward_max[TestItems::ARMOR] = 10.0f;
    return resource_reward_max;
  }

  AgentConfig create_test_agent_config() {
    return AgentConfig(0,                                  // type_id
                       "agent",                            // type_name
                       1,                                  // group_id
                       "test_group",                       // group_name
                       100,                                // freeze_duration
                       0.0f,                               // action_failure_penalty
                       create_test_resource_limits(),      // resource_limits
                       create_test_rewards(),              // resource_rewards
                       create_test_resource_reward_max(),  // resource_reward_max
                       {},                                 // stat_rewards
                       {},                                 // stat_reward_max
                       0.0f,                               // group_reward_pct
                       {});                                // initial_inventory
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

  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test adding items
  int delta = agent->update_inventory(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 5);
  EXPECT_FLOAT_EQ(agent_reward, 0.625f);  // 5 * 0.125

  // Test removing items
  delta = agent->update_inventory(TestItems::ORE, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 3);
  EXPECT_FLOAT_EQ(agent_reward, 0.375f);  // 3 * 0.125

  // Test hitting zero
  delta = agent->update_inventory(TestItems::ORE, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  // check that the item is not in the inventory
  EXPECT_EQ(agent->inventory.find(TestItems::ORE), agent->inventory.end());
  EXPECT_FLOAT_EQ(agent_reward, 0.0f);

  // Test hitting resource_limits limit
  agent->update_inventory(TestItems::ORE, 30);
  delta = agent->update_inventory(TestItems::ORE, 50);  // resource_limits is 50
  EXPECT_EQ(delta, 20);                                 // Should only add up to resource_limits
  EXPECT_EQ(agent->inventory[TestItems::ORE], 50);
  EXPECT_FLOAT_EQ(agent_reward, 6.25f);  // 50 * 0.125
}

// Test for reward capping behavior with a lower cap to actually hit it
TEST_F(MettaGridCppTest, AgentInventoryUpdate_RewardCappingBehavior) {
  // Create a custom config with a lower ore reward cap that we can actually hit
  auto resource_limits = create_test_resource_limits();
  auto rewards = create_test_rewards();

  // Set a lower cap for ORE so we can actually test capping
  std::map<uint8_t, RewardType> resource_reward_max;
  resource_reward_max[TestItems::ORE] = 2.0f;  // Cap at 2.0 instead of 10.0

  AgentConfig agent_cfg(
      0, "agent", 1, "test_group", 100, 0.0f, resource_limits, rewards, resource_reward_max, {}, {}, 0.0f, {});

  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test 1: Add items up to the cap
  // 16 ORE * 0.125 = 2.0 (exactly at cap)
  int delta = agent->update_inventory(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 16);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Test 2: Add more items beyond the cap
  // 32 ORE * 0.125 = 4.0, but capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 32);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Still capped at 2.0

  // Test 3: Remove some items while still over cap
  // 24 ORE * 0.125 = 3.0, but still capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, -8);
  EXPECT_EQ(delta, -8);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 24);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Should remain at cap

  // Test 4: Remove enough items to go below cap
  // 12 ORE * 0.125 = 1.5 (now below cap)
  delta = agent->update_inventory(TestItems::ORE, -12);
  EXPECT_EQ(delta, -12);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 12);
  EXPECT_FLOAT_EQ(agent_reward, 1.5f);  // Now tracking actual value

  // Test 5: Add items again, but not enough to hit cap
  // 14 ORE * 0.125 = 1.75 (still below cap)
  delta = agent->update_inventory(TestItems::ORE, 2);
  EXPECT_EQ(delta, 2);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 14);
  EXPECT_FLOAT_EQ(agent_reward, 1.75f);

  // Test 6: Add items to go over cap again
  // 20 ORE * 0.125 = 2.5, but capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, 6);
  EXPECT_EQ(delta, 6);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 20);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);
}

// Test multiple item types with different caps
TEST_F(MettaGridCppTest, AgentInventoryUpdate_MultipleItemCaps) {
  auto resource_limits = create_test_resource_limits();
  auto rewards = create_test_rewards();

  // Set different caps for different items
  std::map<uint8_t, RewardType> resource_reward_max;
  resource_reward_max[TestItems::ORE] = 2.0f;     // Low cap for ORE
  resource_reward_max[TestItems::HEART] = 30.0f;  // Cap for HEART
  // LASER and ARMOR have no caps

  AgentConfig agent_cfg(
      0, "agent", 1, "test_group", 100, 0.0f, resource_limits, rewards, resource_reward_max, {}, {}, 0.0f, {});

  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add ORE beyond its cap
  agent->update_inventory(TestItems::ORE, 50);  // 50 * 0.125 = 6.25, capped at 2.0
  EXPECT_EQ(agent->inventory[TestItems::ORE], 50);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Add HEART up to its cap
  agent->update_inventory(TestItems::HEART, 30);  // 30 * 1.0 = 30.0
  EXPECT_EQ(agent->inventory[TestItems::HEART], 30);
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // 2.0 + 30.0

  // Add more HEART beyond its cap
  agent->update_inventory(TestItems::HEART, 10);  // 40 * 1.0 = 40.0, capped at 30.0
  EXPECT_EQ(agent->inventory[TestItems::HEART], 40);
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // Still 2.0 + 30.0

  // Remove some ORE (still over cap)
  agent->update_inventory(TestItems::ORE, -10);  // 40 * 0.125 = 5.0, still capped at 2.0
  EXPECT_EQ(agent->inventory[TestItems::ORE], 40);
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // No change

  // Remove ORE to go below cap
  agent->update_inventory(TestItems::ORE, -35);  // 5 * 0.125 = 0.625
  EXPECT_EQ(agent->inventory[TestItems::ORE], 5);
  EXPECT_FLOAT_EQ(agent_reward, 30.625f);  // 0.625 + 30.0

  // Remove HEART to go below its cap
  agent->update_inventory(TestItems::HEART, -15);  // 25 * 1.0 = 25.0
  EXPECT_EQ(agent->inventory[TestItems::HEART], 25);
  EXPECT_FLOAT_EQ(agent_reward, 25.625f);  // 0.625 + 25.0
}

// Test edge case: going to zero
TEST_F(MettaGridCppTest, AgentInventoryUpdate_RewardToZero) {
  auto resource_limits = create_test_resource_limits();
  auto rewards = create_test_rewards();

  std::map<uint8_t, RewardType> resource_reward_max;
  resource_reward_max[TestItems::ORE] = 2.0f;

  AgentConfig agent_cfg(
      0, "agent", 1, "test_group", 100, 0.0f, resource_limits, rewards, resource_reward_max, {}, {}, 0.0f, {});

  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add items beyond cap
  agent->update_inventory(TestItems::ORE, 50);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 50);
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Remove all items
  agent->update_inventory(TestItems::ORE, -50);
  // When inventory goes to zero, the item should be removed from the map
  EXPECT_EQ(agent->inventory.find(TestItems::ORE), agent->inventory.end());
  EXPECT_FLOAT_EQ(agent_reward, 0.0f);
}

// ==================== Grid Tests ====================

TEST_F(MettaGridCppTest, GridCreation) {
  Grid grid(5, 10);  // row/height, col/width

  EXPECT_EQ(grid.width, 10);
  EXPECT_EQ(grid.height, 5);
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
  auto agent_at_location = grid.object_at(GridLocation(2, 3, GridLayer::AgentLayer));
  EXPECT_EQ(agent_at_location, agent);
}

// ==================== Action Tests ====================

TEST_F(MettaGridCppTest, AttackAction) {
  Grid grid(10, 10);

  // Create a minimal GameConfig for testing
  GameConfig game_config;
  game_config.allow_diagonals = false;  // Test with cardinal directions only

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
  EXPECT_EQ(attacker->orientation, Orientation::North);

  // Create attack action handler
  AttackActionConfig attack_cfg({{TestItems::LASER, 1}}, {{TestItems::LASER, 1}}, {{TestItems::ARMOR, 3}});
  Attack attack(attack_cfg, &game_config);
  std::mt19937 rng(42);
  attack.init(&grid, &rng);

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
  ConverterConfig generator_cfg(TestItems::CONVERTER,     // type_id
                                "generator",              // type_name
                                {{TestItems::ORE, 1}},    // input_resources
                                {{TestItems::ARMOR, 1}},  // output_resources
                                0,                        // max_output
                                -1,                       // max_conversions
                                1,                        // conversion_ticks
                                10,                       // cooldown
                                0,                        // initial_resource_count
                                0,                        // color
                                false);                   // recipe_details_obs
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(TestItems::ORE, 1);
  agent->update_inventory(TestItems::HEART, 1);

  // Create put_items action handler
  ActionConfig put_cfg({}, {});
  PutRecipeItems put(put_cfg);
  std::mt19937 rng(42);
  put.init(&grid, &rng);

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
  ConverterConfig generator_cfg(TestItems::CONVERTER,     // type_id
                                "generator",              // type_name
                                {{TestItems::ORE, 1}},    // input_resources
                                {{TestItems::ARMOR, 1}},  // output_resources
                                1,                        // max_output
                                -1,                       // max_conversions
                                1,                        // conversion_ticks
                                10,                       // cooldown
                                1,                        // initial_items
                                0,                        // color
                                false);                   // recipe_details_obs
  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(TestItems::ORE, 1);

  // Create get_output action handler
  ActionConfig get_cfg({}, {});
  GetOutput get(get_cfg);
  std::mt19937 rng(42);
  get.init(&grid, &rng);

  // Test getting output
  bool success = get.handle_action(agent->id, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[TestItems::ORE], 1);        // Still have ore
  EXPECT_EQ(agent->inventory[TestItems::ARMOR], 1);      // Also have armor
  EXPECT_EQ(generator->inventory[TestItems::ARMOR], 0);  // Generator gave away its armor
}

// ==================== Action Tracking ====================

TEST_F(MettaGridCppTest, ActionTracking) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  ActionConfig noop_cfg({}, {});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 0.0f);
  noop.handle_action(agent->id, 0);  // count 1, max 1
  EXPECT_EQ(agent->location.r, 5);
  EXPECT_EQ(agent->location.c, 5);
  EXPECT_EQ(agent->prev_location.r, 5);
  EXPECT_EQ(agent->prev_location.c, 5);
  EXPECT_EQ(agent->prev_action_name, "noop");
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 1.0f);
  agent->location.r = 6;
  agent->location.c = 6;
  noop.handle_action(agent->id, 0);  // count 0, max 1
  EXPECT_EQ(agent->location.r, 6);
  EXPECT_EQ(agent->location.c, 6);
  EXPECT_EQ(agent->prev_location.r, 6);
  EXPECT_EQ(agent->prev_location.c, 6);
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 1.0f);
  noop.handle_action(agent->id, 0);  // count 1, max 1
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 1.0f);
  noop.handle_action(agent->id, 0);  // count 2, max 2
  noop.handle_action(agent->id, 0);  // count 3, max 3
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 3.0f);
  agent->location.r = 7;
  agent->location.c = 7;
  noop.handle_action(agent->id, 0);  // count 0, max 3
  EXPECT_EQ(agent->location.r, 7);
  EXPECT_EQ(agent->location.c, 7);
  EXPECT_EQ(agent->prev_location.r, 7);
  EXPECT_EQ(agent->prev_location.c, 7);
  noop.handle_action(agent->id, 0);  // count 1, max 3
  noop.handle_action(agent->id, 0);  // count 2, max 3
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 3.0f);
  noop.handle_action(agent->id, 0);  // count 3, max 3
  noop.handle_action(agent->id, 0);  // count 4, max 4
  EXPECT_FLOAT_EQ(agent->stats.to_dict()["status.max_steps_without_motion"], 4.0f);
}

// ==================== Fractional Consumption Tests ====================

TEST_F(MettaGridCppTest, FractionalConsumptionProbability) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with fractional consumption (0.5)
  // Required resources must be at least ceil(consumed) = 1
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.5f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action multiple times
  for (int i = 0; i < 10; i++) {
    noop.handle_action(agent->id, 0);
  }

  // With 0.5 probability, exactly 4 ore should be consumed (10 - 4 = 6 remaining)
  int final_ore = agent->inventory.count(TestItems::ORE) > 0 ? agent->inventory[TestItems::ORE] : 0;
  EXPECT_EQ(final_ore, 6);

  // Test that action fails when inventory is empty
  AgentConfig poor_cfg = create_test_agent_config();
  // Don't set initial_inventory so the agent starts with nothing
  Agent* poor_agent = new Agent(2, 1, poor_cfg);
  float poor_reward = 0.0f;
  poor_agent->init(&poor_reward);
  grid.add_object(poor_agent);

  bool success = noop.handle_action(poor_agent->id, 0);
  EXPECT_FALSE(success);  // Should fail due to insufficient resources
}

TEST_F(MettaGridCppTest, FractionalConsumptionWithOverflow) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 5;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with fractional consumption (1.5)
  // Required resources must be at least ceil(consumed) = 2
  ActionConfig noop_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.5f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  bool success = noop.handle_action(agent->id, 0);
  EXPECT_TRUE(success);  // Should succeed as we have enough resources

  // With 1.5, should consume either 1 or 2 units
  int final_ore = agent->inventory.count(TestItems::ORE) > 0 ? agent->inventory[TestItems::ORE] : 0;
  EXPECT_TRUE(final_ore == 3 || final_ore == 4);
}

TEST_F(MettaGridCppTest, FractionalConsumptionRequiresCeiledInventory) {
  Grid grid(3, 3);

  // Create agent with only 1 resource
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 1;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with fractional consumption (1.5) - requires ceil(1.5) = 2
  ActionConfig noop_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.5f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  bool success = noop.handle_action(agent->id, 0);
  EXPECT_FALSE(success);  // Should fail as we only have 1 but need ceil(1.5) = 2

  // Verify inventory unchanged
  EXPECT_EQ(agent->inventory[TestItems::ORE], 1);
}

TEST_F(MettaGridCppTest, FractionalConsumptionInvalidRequirements) {
  // Test that ActionHandler constructor throws when required < ceil(consumed)
  ActionConfig invalid_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 1.5f}});

  // This should throw because required (1) < ceil(consumed) (2)
  EXPECT_THROW({ Noop noop(invalid_cfg); }, std::runtime_error);
}

TEST_F(MettaGridCppTest, FractionalConsumptionZero) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with zero consumption
  // Required resources must be at least ceil(0.0) = 0
  ActionConfig noop_cfg({{TestItems::ORE, 0}}, {{TestItems::ORE, 0.0f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action multiple times - should never consume
  for (int i = 0; i < 10; i++) {
    bool success = noop.handle_action(agent->id, 0);
    EXPECT_TRUE(success);
  }

  // Verify no resources consumed
  EXPECT_EQ(agent->inventory[TestItems::ORE], 10);
}

TEST_F(MettaGridCppTest, FractionalConsumptionInteger) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with integer consumption (2.0)
  ActionConfig noop_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 2.0f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action 3 times - should always consume exactly 2
  for (int i = 0; i < 3; i++) {
    bool success = noop.handle_action(agent->id, 0);
    EXPECT_TRUE(success);
  }

  // Verify exactly 6 resources consumed (3 * 2)
  EXPECT_EQ(agent->inventory[TestItems::ORE], 4);
}

TEST_F(MettaGridCppTest, FractionalConsumptionSmallFraction) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 20;  // Enough for test
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with small fractional consumption (0.1)
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.1f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action many times - only count successful actions
  int consumed = 0;
  int successful_actions = 0;
  for (int i = 0; i < 100; i++) {
    int before = agent->inventory[TestItems::ORE];
    bool success = noop.handle_action(agent->id, 0);
    if (success) {
      successful_actions++;
      int after = agent->inventory[TestItems::ORE];
      consumed += (before - after);
    }
  }

  EXPECT_EQ(successful_actions, 100);  // All 100 attempts succeed (only 16 consumed)
  EXPECT_EQ(consumed, 16);             // Exactly 16 ore consumed
}

TEST_F(MettaGridCppTest, FractionalConsumptionLargeFraction) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 100;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with large fractional consumption (0.9)
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.9f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action many times - only count successful actions
  int consumed = 0;
  int successful_actions = 0;
  for (int i = 0; i < 200; i++) {
    int before = agent->inventory[TestItems::ORE];
    bool success = noop.handle_action(agent->id, 0);
    if (success) {
      successful_actions++;
      int after = agent->inventory[TestItems::ORE];
      consumed += (before - after);
    }
  }

  EXPECT_EQ(successful_actions, 114);  // Exactly 114 successful actions before running out
  EXPECT_EQ(consumed, 100);            // All 100 ore consumed
}

TEST_F(MettaGridCppTest, FractionalConsumptionMultipleResources) {
  Grid grid(3, 3);

  // Create agent with multiple resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 50;
  agent_cfg.initial_inventory[TestItems::LASER] = 50;
  agent_cfg.initial_inventory[TestItems::ARMOR] = 50;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with different fractional consumptions
  ActionConfig noop_cfg({{TestItems::ORE, 2}, {TestItems::LASER, 1}, {TestItems::ARMOR, 3}},
                        {{TestItems::ORE, 1.5f}, {TestItems::LASER, 0.25f}, {TestItems::ARMOR, 2.75f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Execute action multiple times
  for (int i = 0; i < 10; i++) {
    bool success = noop.handle_action(agent->id, 0);
    EXPECT_TRUE(success);
  }

  int ore_left = agent->inventory[TestItems::ORE];
  int laser_left = agent->inventory[TestItems::LASER];
  int armor_left = agent->inventory[TestItems::ARMOR];

  EXPECT_EQ(ore_left, 33);

  EXPECT_EQ(laser_left, 48);

  EXPECT_EQ(armor_left, 24);
}

TEST_F(MettaGridCppTest, FractionalConsumptionAttackAction) {
  // This test verifies that fractional consumption works with attack actions
  // We'll do a simple test with a few attacks rather than a complex loop

  Grid grid(10, 10);
  GameConfig game_config;

  // Create attacker with lasers
  AgentConfig attacker_cfg = create_test_agent_config();
  attacker_cfg.group_name = "red";
  Agent* attacker = new Agent(2, 0, attacker_cfg);

  // Create target
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

  // Give attacker 10 lasers
  attacker->update_inventory(TestItems::LASER, 10);
  // Give target some hearts to rob
  target->update_inventory(TestItems::HEART, 5);

  // Create attack action with fractional laser consumption (0.5 per attack)
  AttackActionConfig attack_cfg({{TestItems::LASER, 1}}, {{TestItems::LASER, 0.5f}}, {});
  Attack attack(attack_cfg, &game_config);
  std::mt19937 rng(42);
  attack.init(&grid, &rng);

  // Track consumption over multiple attacks
  int total_consumed = 0;
  int successful_attacks = 0;

  // Do 10 attacks
  for (int i = 0; i < 10; i++) {
    int before = attacker->inventory[TestItems::LASER];
    bool success = attack.handle_action(attacker->id, 5);  // Attack directly in front
    if (success) {
      successful_attacks++;
      int after = attacker->inventory[TestItems::LASER];
      total_consumed += (before - after);
    }
  }

  EXPECT_EQ(successful_attacks, 10);  // All 10 attacks succeed with initial 10 lasers
  EXPECT_EQ(total_consumed, 4);       // Exactly 4 lasers consumed from 10 attacks
}

TEST_F(MettaGridCppTest, FractionalConsumptionChangeGlyphAction) {
  Grid grid(3, 3);

  // Create agent with resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 30;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create change glyph action with fractional consumption (1.25)
  ChangeGlyphActionConfig glyph_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.25f}}, 4);
  ChangeGlyph change_glyph(glyph_cfg);
  std::mt19937 rng(42);
  change_glyph.init(&grid, &rng);

  // Change glyph multiple times
  int changes = 0;
  ObservationType initial_glyph = agent->glyph;
  while (agent->inventory[TestItems::ORE] >= 2) {
    bool success = change_glyph.handle_action(agent->id, (initial_glyph + 1) % 4);
    if (!success) break;
    changes++;
    if (changes > 30) break;  // Safety limit
  }

  // With 1.25 consumption and 30 initial ore, we expect around 24 changes
  // (30 / 1.25 = 24)
  EXPECT_GE(changes, 22);
  EXPECT_LE(changes, 26);
}

TEST_F(MettaGridCppTest, FractionalConsumptionBoundaryValues) {
  Grid grid(3, 3);

  // Create agent with exact boundary amount
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 1;
  Agent* agent = new Agent(1, 1, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Test with 0.99 consumption (almost always consumes 1)
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.99f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Should succeed once then likely fail
  bool first_success = noop.handle_action(agent->id, 0);
  EXPECT_TRUE(first_success);

  // Very high chance we consumed the resource (99%)
  bool second_success = noop.handle_action(agent->id, 0);
  // This will almost certainly fail (99% chance we're out of resources)
  if (agent->inventory[TestItems::ORE] == 0) {
    EXPECT_FALSE(second_success);
  }
}

TEST_F(MettaGridCppTest, FractionalConsumptionDeterministicWithSameSeed) {
  Grid grid1(3, 3);
  Grid grid2(3, 3);

  // Create two identical setups
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 100;

  Agent* agent1 = new Agent(1, 1, agent_cfg);
  float reward1 = 0.0f;
  agent1->init(&reward1);
  grid1.add_object(agent1);

  Agent* agent2 = new Agent(1, 1, agent_cfg);
  float reward2 = 0.0f;
  agent2->init(&reward2);
  grid2.add_object(agent2);

  // Create identical actions with same seed
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.33f}});
  Noop noop1(noop_cfg);
  Noop noop2(noop_cfg);

  std::mt19937 rng1(42);
  std::mt19937 rng2(42);
  noop1.init(&grid1, &rng1);
  noop2.init(&grid2, &rng2);

  // Execute same sequence on both
  for (int i = 0; i < 50; i++) {
    noop1.handle_action(agent1->id, 0);
    noop2.handle_action(agent2->id, 0);
  }

  // Should have identical results with same seed
  EXPECT_EQ(agent1->inventory[TestItems::ORE], agent2->inventory[TestItems::ORE]);
}

// ==================== Event System Tests ====================

TEST_F(MettaGridCppTest, EventManager) {
  Grid grid(10, 10);
  EventManager event_manager;

  // Test that event manager can be initialized
  // (This is a basic test - more complex event testing would require more setup)
  EXPECT_NO_THROW(event_manager.process_events(1));
}
