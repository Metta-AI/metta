#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "mettagrid/actions/attack.hpp"
#include "mettagrid/actions/get_output.hpp"
#include "mettagrid/actions/noop.hpp"
#include "mettagrid/actions/put_recipe_items.hpp"
#include "mettagrid/actions/resource_mod.hpp"
#include "mettagrid/event.hpp"
#include "mettagrid/grid.hpp"
#include "mettagrid/mettagrid_config.hpp"
#include "mettagrid/objects/agent.hpp"
#include "mettagrid/objects/constants.hpp"
#include "mettagrid/objects/converter.hpp"
#include "mettagrid/objects/wall.hpp"
#include "mettagrid/types.hpp"

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
    resource_limits[TestItems::HEART] = 100;  // Increased to allow testing of fractional modifications
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
  std::mt19937 rng(42);

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
  std::mt19937 rng(42);

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
  std::mt19937 rng(42);

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
  std::mt19937 rng(42);

  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  ActionConfig noop_cfg({}, {});
  Noop noop(noop_cfg);
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

// ==================== Event System Tests ====================

TEST_F(MettaGridCppTest, EventManager) {
  Grid grid(10, 10);
  EventManager event_manager;

  // Test that event manager can be initialized
  // (This is a basic test - more complex event testing would require more setup)
  EXPECT_NO_THROW(event_manager.process_events(1));
}

// ==================== ResourceMod Tests ====================

TEST_F(MettaGridCppTest, ResourceModBasicConsumption) {
  Grid grid(10, 10);
  std::mt19937 rng(42);

  // Create agent with resources
  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Give agent some resources
  agent->update_inventory(TestItems::ORE, 10);
  agent->update_inventory(TestItems::HEART, 20);

  // Create ResourceModActionConfig that consumes resources deterministically
  ResourceModActionConfig cfg({},                        // required_resources
                              {},                        // consumed_resources
                              {{TestItems::ORE, 1.0f}},  // consumes - 100% chance to consume 1 ore
                              {},                        // modifies
                              false,                     // scales
                              0,                         // agent_radius
                              0);                        // converter_radius

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  // Execute action
  bool success = action.handle_action(agent->id, 0);
  EXPECT_TRUE(success);

  // Check that ore was consumed
  EXPECT_EQ(agent->inventory[TestItems::ORE], 9);
  EXPECT_EQ(agent->inventory[TestItems::HEART], 20);  // Unchanged
}

TEST_F(MettaGridCppTest, ResourceModNearbyAgents) {
  Grid grid(10, 10);
  std::mt19937 rng(42);

  // Create multiple agents
  AgentConfig agent_cfg = create_test_agent_config();
  Agent* actor = new Agent(5, 5, agent_cfg);
  Agent* nearby = new Agent(5, 6, agent_cfg);  // Distance 1
  Agent* far = new Agent(5, 8, agent_cfg);     // Distance 3

  float reward1 = 0.0f, reward2 = 0.0f, reward3 = 0.0f;
  actor->init(&reward1);
  nearby->init(&reward2);
  far->init(&reward3);

  grid.add_object(actor);
  grid.add_object(nearby);
  grid.add_object(far);

  // Give resources
  actor->update_inventory(TestItems::ORE, 10);
  nearby->update_inventory(TestItems::HEART, 10);
  far->update_inventory(TestItems::HEART, 10);

  // Create ResourceMod that affects agents within radius 2
  ResourceModActionConfig cfg({},                          // required_resources
                              {},                          // consumed_resources
                              {{TestItems::ORE, 1.0f}},    // consumes 1 ore from actor
                              {{TestItems::HEART, 1.0f}},  // adds 1 heart to nearby agents
                              false,                       // scales
                              2,                           // agent_radius - affects agents within distance 2
                              0);                           // converter_radius

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  // Execute action
  bool success = action.handle_action(actor->id, 0);
  EXPECT_TRUE(success);

  // Check consumption
  EXPECT_EQ(actor->inventory[TestItems::ORE], 9);  // Consumed 1 ore

  // Check modifications
  EXPECT_EQ(actor->inventory[TestItems::HEART], 1);    // Actor gets +1 (distance 0, within radius 2)
  EXPECT_EQ(nearby->inventory[TestItems::HEART], 11);  // Nearby gets +1 (distance 1, within radius 2)
  EXPECT_EQ(far->inventory[TestItems::HEART], 10);     // Far unchanged (distance 3, outside radius 2)
}

TEST_F(MettaGridCppTest, ResourceModScaling) {
  Grid grid(10, 10);
  std::mt19937 rng(42);

  // Create agents close together
  AgentConfig agent_cfg = create_test_agent_config();
  Agent* actor = new Agent(5, 5, agent_cfg);
  Agent* target1 = new Agent(5, 6, agent_cfg);  // Distance 1
  Agent* target2 = new Agent(6, 5, agent_cfg);  // Distance 1

  float reward1 = 0.0f, reward2 = 0.0f, reward3 = 0.0f;
  actor->init(&reward1);
  target1->init(&reward2);
  target2->init(&reward3);

  grid.add_object(actor);
  grid.add_object(target1);
  grid.add_object(target2);

  actor->update_inventory(TestItems::ORE, 10);

  // Create ResourceMod with scaling enabled
  // With 3 targets total, each has 0.99/3 = 0.33 probability
  ResourceModActionConfig cfg({},                           // required_resources
                              {},                           // consumed_resources
                              {{TestItems::ORE, 1.0f}},     // consumes
                              {{TestItems::HEART, 0.99f}},  // modifies - scaled by 3 targets = 0.33 each
                              true,                         // scales enabled
                              1,                            // agent_radius
                              0);                           // converter_radius

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  // Execute action multiple times to test probabilistic behavior
  int actor_gains = 0;
  int target1_gains = 0;
  int target2_gains = 0;

  // Run 100 iterations with different seeds
  for (int i = 0; i < 100; i++) {
    // Reset inventories
    actor->inventory[TestItems::ORE] = 10;
    actor->inventory[TestItems::HEART] = 0;
    target1->inventory[TestItems::HEART] = 0;
    target2->inventory[TestItems::HEART] = 0;

    // Use different seed each time
    std::mt19937 test_rng(42 + i);
    action.init(&grid, &test_rng);

    action.handle_action(actor->id, 0);

    actor_gains += actor->inventory[TestItems::HEART];
    target1_gains += target1->inventory[TestItems::HEART];
    target2_gains += target2->inventory[TestItems::HEART];
  }

  // With 33% probability per target, expect around 33 gains per 100 runs
  // Allow reasonable variance (20-45 range)
  EXPECT_GE(actor_gains, 20);
  EXPECT_LE(actor_gains, 45);
  EXPECT_GE(target1_gains, 20);
  EXPECT_LE(target1_gains, 45);
  EXPECT_GE(target2_gains, 20);
  EXPECT_LE(target2_gains, 45);
}

TEST_F(MettaGridCppTest, ResourceModFractionalAmounts) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  agent->update_inventory(TestItems::ORE, 100);
  agent->update_inventory(TestItems::HEART, 50);

  // Test fractional consumption (30% chance)
  ResourceModActionConfig cfg({},                          // required_resources
                              {},                          // consumed_resources
                              {{TestItems::ORE, 0.3f}},    // 30% chance to consume 1
                              {{TestItems::HEART, 0.7f}},  // 70% chance to add 1
                              false,                       // scales
                              1,                           // agent_radius (includes self)
                              0);                          // converter_radius

  // Run multiple times to test probabilistic behavior
  int ore_consumed_count = 0;
  int heart_gained_count = 0;

  // Store rngs and actions to keep them alive throughout the test
  std::vector<std::unique_ptr<std::mt19937>> rngs;
  std::vector<std::unique_ptr<ResourceMod>> actions;

  for (int i = 0; i < 100; i++) {
    agent->inventory[TestItems::ORE] = 100;
    agent->inventory[TestItems::HEART] = 50;

    // Create and store rng and action
    rngs.push_back(std::make_unique<std::mt19937>(42 + i));
    actions.push_back(std::make_unique<ResourceMod>(cfg));
    actions.back()->init(&grid, rngs.back().get());

    actions.back()->handle_action(agent->id, 0);

    if (agent->inventory[TestItems::ORE] < 100) {
      ore_consumed_count++;
    }
    if (agent->inventory[TestItems::HEART] > 50) {
      heart_gained_count++;
    }
  }

  // Expect approximately 30% consumption rate and 70% gain rate
  // Allow variance of ±15%
  EXPECT_GE(ore_consumed_count, 15);  // At least 15%
  EXPECT_LE(ore_consumed_count, 45);  // At most 45%
  EXPECT_GE(heart_gained_count, 55);  // At least 55%
  EXPECT_LE(heart_gained_count, 85);  // At most 85%
}

TEST_F(MettaGridCppTest, ResourceModNegativeModifications) {
  Grid grid(10, 10);
  std::mt19937 rng(42);

  AgentConfig agent_cfg = create_test_agent_config();
  Agent* actor = new Agent(5, 5, agent_cfg);
  Agent* target = new Agent(5, 6, agent_cfg);

  float reward1 = 0.0f, reward2 = 0.0f;
  actor->init(&reward1);
  target->init(&reward2);

  grid.add_object(actor);
  grid.add_object(target);

  actor->update_inventory(TestItems::ORE, 10);
  actor->update_inventory(TestItems::HEART, 20);
  target->update_inventory(TestItems::HEART, 20);

  // Create ResourceMod that damages nearby agents
  ResourceModActionConfig cfg({},                           // required_resources
                              {},                           // consumed_resources
                              {{TestItems::ORE, 1.0f}},     // consumes 1 ore to cast
                              {{TestItems::HEART, -1.0f}},  // removes 1 heart from targets
                              false,                        // scales
                              1,                            // agent_radius
                              0);                           // converter_radius

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  bool success = action.handle_action(actor->id, 0);
  EXPECT_TRUE(success);

  // Check consumption
  EXPECT_EQ(actor->inventory[TestItems::ORE], 9);

  // Check negative modifications (both actors lose 1 heart)
  EXPECT_EQ(actor->inventory[TestItems::HEART], 19);
  EXPECT_EQ(target->inventory[TestItems::HEART], 19);
}

TEST_F(MettaGridCppTest, ResourceModAtomicity) {
  Grid grid(10, 10);
  std::mt19937 rng(42);

  AgentConfig agent_cfg = create_test_agent_config();
  Agent* agent = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Give agent limited resources
  agent->update_inventory(TestItems::ORE, 5);
  agent->update_inventory(TestItems::HEART, 5);
  // No ARMOR (TestItems::ARMOR)

  // Create ResourceMod that tries to consume multiple resources
  // This should fail atomically since we don't have ARMOR
  ResourceModActionConfig cfg({},  // required_resources
                              {},  // consumed_resources
                              {
                                  {TestItems::ORE, 1.0f},    // have this
                                  {TestItems::HEART, 1.0f},  // have this
                                  {TestItems::ARMOR, 1.0f}   // DON'T have this - should cause atomic failure
                              },
                              {},     // modifies
                              false,  // scales
                              0,      // agent_radius
                              0);     // converter_radius

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  bool success = action.handle_action(agent->id, 0);
  EXPECT_FALSE(success);  // Should fail due to missing ARMOR

  // CRITICAL: Check that NO resources were consumed (atomicity)
  EXPECT_EQ(agent->inventory[TestItems::ORE], 5);    // Unchanged
  EXPECT_EQ(agent->inventory[TestItems::HEART], 5);  // Unchanged
  EXPECT_EQ(agent->inventory[TestItems::ARMOR], 0);  // Still 0
}

TEST_F(MettaGridCppTest, ResourceModConverters) {
  Grid grid(10, 10);
  std::mt19937 rng(42);
  EventManager event_manager;

  // Create agent
  AgentConfig agent_cfg = create_test_agent_config();
  Agent* actor = new Agent(5, 5, agent_cfg);
  float agent_reward = 0.0f;
  actor->init(&agent_reward);
  grid.add_object(actor);

  // Create converters at different distances
  ConverterConfig converter_cfg(TestItems::CONVERTER,  // type_id
                                "converter",           // type_name
                                {},                    // input_resources (empty - just for storage)
                                {},                    // output_resources (empty)
                                -1,                    // max_output
                                -1,                    // max_conversions
                                0,                     // conversion_ticks
                                0,                     // cooldown
                                0,                     // initial_items
                                0,                     // color
                                false);                // recipe_details_obs

  Converter* near_converter = new Converter(5, 7, converter_cfg);  // Distance 2
  Converter* far_converter = new Converter(5, 9, converter_cfg);   // Distance 4

  grid.add_object(near_converter);
  grid.add_object(far_converter);
  near_converter->set_event_manager(&event_manager);
  far_converter->set_event_manager(&event_manager);

  // Give agent resources
  actor->update_inventory(TestItems::ORE, 10);

  // Create ResourceMod that affects converters within radius 2
  ResourceModActionConfig cfg({},                          // required_resources
                              {},                          // consumed_resources
                              {{TestItems::ORE, 1.0f}},    // consumes
                              {{TestItems::HEART, 1.0f}},  // adds 1 heart to targets
                              false,                       // scales
                              0,                           // agent_radius - don't affect agents
                              2);                          // converter_radius - affect converters within distance 2

  ResourceMod action(cfg);
  action.init(&grid, &rng);

  bool success = action.handle_action(actor->id, 0);
  EXPECT_TRUE(success);

  // Check consumption
  EXPECT_EQ(actor->inventory[TestItems::ORE], 9);

  // Check modifications
  EXPECT_EQ(near_converter->inventory[TestItems::HEART], 1);  // Distance 2, within radius
  EXPECT_EQ(far_converter->inventory[TestItems::HEART], 0);   // Distance 4, outside radius
}
