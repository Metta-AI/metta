#include <gtest/gtest.h>

#include <array>
#include <random>
#include <utility>

#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "actions/get_output.hpp"
#include "actions/noop.hpp"
#include "actions/put_recipe_items.hpp"
#include "actions/resource_mod.hpp"
#include "config/mettagrid_config.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"
#include "objects/inventory_config.hpp"
#include "objects/production_handler.hpp"
#include "objects/wall.hpp"

// Test-specific inventory item type constants
namespace TestItems {
constexpr uint8_t ORE = 0;
constexpr uint8_t LASER = 1;
constexpr uint8_t ARMOR = 2;
constexpr uint8_t HEART = 3;
constexpr uint8_t CONVERTER = 4;
}  // namespace TestItems

namespace TestItemStrings {
const char ORE[] = "ore_red";
const char LASER[] = "laser";
const char ARMOR[] = "armor";
const char HEART[] = "heart";
}  // namespace TestItemStrings

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
  InventoryConfig create_test_inventory_config() {
    InventoryConfig inventory_config;
    inventory_config.limits = {
        {{TestItems::ORE}, 50},
        {{TestItems::LASER}, 50},
        {{TestItems::ARMOR}, 50},
        {{TestItems::HEART}, 50},
    };
    return inventory_config;
  }

  std::unordered_map<std::string, RewardType> create_test_stats_rewards() {
    std::unordered_map<std::string, RewardType> rewards;
    rewards[std::string(TestItemStrings::ORE) + ".amount"] = TestRewards::ORE;
    rewards[std::string(TestItemStrings::LASER) + ".amount"] = TestRewards::LASER;
    rewards[std::string(TestItemStrings::ARMOR) + ".amount"] = TestRewards::ARMOR;
    rewards[std::string(TestItemStrings::HEART) + ".amount"] = TestRewards::HEART;
    return rewards;
  }

  // Helper function to create test stats_reward_max map
  std::unordered_map<std::string, RewardType> create_test_stats_reward_max() {
    std::unordered_map<std::string, RewardType> stats_reward_max;
    stats_reward_max[std::string(TestItemStrings::ORE) + ".amount"] = 10.0f;
    stats_reward_max[std::string(TestItemStrings::LASER) + ".amount"] = 10.0f;
    stats_reward_max[std::string(TestItemStrings::ARMOR) + ".amount"] = 10.0f;
    return stats_reward_max;
  }

  std::vector<std::string> create_test_resource_names() {
    return {TestItemStrings::ORE, TestItemStrings::LASER, TestItemStrings::ARMOR, TestItemStrings::HEART};
  }

  AgentConfig create_test_agent_config() {
    return AgentConfig(0,                               // type_id
                       "agent",                         // type_name
                       1,                               // group_id
                       "test_group",                    // group_name
                       100,                             // freeze_duration
                       0.0f,                            // action_failure_penalty
                       create_test_inventory_config(),  // resource_limits
                       create_test_stats_rewards(),     // stats_rewards
                       create_test_stats_reward_max(),  // stats_reward_max
                       0.0f,                            // group_reward_pct
                       {});                             // initial_inventory
  }
};

static void RegisterProductionHandlers(EventManager& event_manager) {
  auto finish_handler = std::make_unique<ProductionHandler>(&event_manager);
  event_manager.event_handlers.insert({EventType::FinishConverting, std::move(finish_handler)});

  auto cooldown_handler = std::make_unique<CoolDownHandler>(&event_manager);
  event_manager.event_handlers.insert({EventType::CoolDown, std::move(cooldown_handler)});
}

// ==================== Agent Tests ====================

TEST_F(MettaGridCppTest, AgentRewards) {
  AgentConfig agent_cfg = create_test_agent_config();
  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));

  // Test reward values
  EXPECT_FLOAT_EQ(agent->stat_rewards[std::string(TestItemStrings::ORE) + ".amount"], 0.125f);
  EXPECT_FLOAT_EQ(agent->stat_rewards[std::string(TestItemStrings::LASER) + ".amount"], 0.0f);
  EXPECT_FLOAT_EQ(agent->stat_rewards[std::string(TestItemStrings::ARMOR) + ".amount"], 0.0f);
  EXPECT_FLOAT_EQ(agent->stat_rewards[std::string(TestItemStrings::HEART) + ".amount"], 1.0f);
}

TEST_F(MettaGridCppTest, AgentRewardsWithAdditionalStatsTracker) {
  // Create agent with reward for chest.hearts.amount
  auto rewards = create_test_stats_rewards();
  rewards["chest.heart.amount"] = 0.1f;

  auto stats_reward_max = create_test_stats_reward_max();
  stats_reward_max["chest.heart.amount"] = 5.0f;

  AgentConfig agent_cfg(
      0, "agent", 1, "test_group", 100, 0.0f, create_test_inventory_config(), rewards, stats_reward_max);
  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));

  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Set up agent's own stats
  agent->stats.set("heart.amount", 5.0f);  // Agent has 5 hearts

  // Create an additional stats tracker (e.g., from game or chest)
  StatsTracker additional_stats(&resource_names);
  additional_stats.set("chest.heart.amount", 10.0f);  // Additional 10 chest hearts

  // Compute rewards without additional tracker
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 5.0f);

  // Compute rewards with additional tracker
  agent->compute_stat_rewards(&additional_stats);
  EXPECT_FLOAT_EQ(agent_reward, 6.0f);  // 5 + 0.1 * 10

  // Test with values that exceed the cap
  additional_stats.set("chest.heart.amount", 100.0f);
  agent->compute_stat_rewards(&additional_stats);
  EXPECT_FLOAT_EQ(agent_reward, 10.0f);

  // Check that they add up if both stats trackers have the same entry (even though we don't expect to use this)
  additional_stats.set("chest.heart.amount", 10.0f);
  agent->stats.set("chest.heart.amount", 10.0f);
  agent->compute_stat_rewards(&additional_stats);
  EXPECT_FLOAT_EQ(agent_reward, 7.0f);  // 5 + 0.1 * 10 + 10
}

TEST_F(MettaGridCppTest, AgentInventoryUpdate) {
  AgentConfig agent_cfg = create_test_agent_config();
  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));

  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test adding items
  int delta = agent->update_inventory(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.625f);  // 5 * 0.125

  // Test removing items
  delta = agent->update_inventory(TestItems::ORE, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 3);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.375f);  // 3 * 0.125

  // Test hitting zero
  delta = agent->update_inventory(TestItems::ORE, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  // check that the item is not in the inventory
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.0f);

  // Test hitting resource_limits limit
  agent->update_inventory(TestItems::ORE, 30);
  delta = agent->update_inventory(TestItems::ORE, 50);  // resource_limits is 50
  EXPECT_EQ(delta, 20);                                 // Should only add up to resource_limits
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 50);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 6.25f);  // 50 * 0.125
}

// Test for reward capping behavior with a lower cap to actually hit it
TEST_F(MettaGridCppTest, AgentInventoryUpdate_RewardCappingBehavior) {
  // Create a custom config with a lower ore reward cap that we can actually hit
  auto inventory_config = create_test_inventory_config();
  auto rewards = create_test_stats_rewards();

  // Set a lower cap for ORE so we can actually test capping
  std::unordered_map<std::string, RewardType> stats_reward_max;
  stats_reward_max[std::string(TestItemStrings::ORE) + ".amount"] = 2.0f;  // Cap at 2.0 instead of 10.0

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0.0f, inventory_config, rewards, stats_reward_max, 0.0f, {});

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test 1: Add items up to the cap
  // 16 ORE * 0.125 = 2.0 (exactly at cap)
  int delta = agent->update_inventory(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 16);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Test 2: Add more items beyond the cap
  // 32 ORE * 0.125 = 4.0, but capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 32);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Still capped at 2.0

  // Test 3: Remove some items while still over cap
  // 24 ORE * 0.125 = 3.0, but still capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, -8);
  EXPECT_EQ(delta, -8);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 24);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Should remain at cap

  // Test 4: Remove enough items to go below cap
  // 12 ORE * 0.125 = 1.5 (now below cap)
  delta = agent->update_inventory(TestItems::ORE, -12);
  EXPECT_EQ(delta, -12);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 12);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 1.5f);  // Now tracking actual value

  // Test 5: Add items again, but not enough to hit cap
  // 14 ORE * 0.125 = 1.75 (still below cap)
  delta = agent->update_inventory(TestItems::ORE, 2);
  EXPECT_EQ(delta, 2);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 14);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 1.75f);

  // Test 6: Add items to go over cap again
  // 20 ORE * 0.125 = 2.5, but capped at 2.0
  delta = agent->update_inventory(TestItems::ORE, 6);
  EXPECT_EQ(delta, 6);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 20);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);
}

// Test multiple item types with different caps
TEST_F(MettaGridCppTest, AgentInventoryUpdate_MultipleItemCaps) {
  auto inventory_config = create_test_inventory_config();
  auto rewards = create_test_stats_rewards();

  // Set different caps for different items
  std::unordered_map<std::string, RewardType> stats_reward_max;
  stats_reward_max[std::string(TestItemStrings::ORE) + ".amount"] = 2.0f;     // Low cap for ORE
  stats_reward_max[std::string(TestItemStrings::HEART) + ".amount"] = 30.0f;  // Cap for HEART
  // LASER and ARMOR have no caps

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0.0f, inventory_config, rewards, stats_reward_max, 0.0f, {});

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add ORE beyond its cap
  agent->update_inventory(TestItems::ORE, 50);  // 50 * 0.125 = 6.25, capped at 2.0
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 50);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Add HEART up to its cap
  agent->update_inventory(TestItems::HEART, 30);  // 30 * 1.0 = 30.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 30);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // 2.0 + 30.0

  // Add more HEART beyond its cap
  agent->update_inventory(TestItems::HEART, 10);  // 40 * 1.0 = 40.0, capped at 30.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 40);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // Still 2.0 + 30.0

  // Remove some ORE (still over cap)
  agent->update_inventory(TestItems::ORE, -10);  // 40 * 0.125 = 5.0, still capped at 2.0
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 40);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // No change

  // Remove ORE to go below cap
  agent->update_inventory(TestItems::ORE, -35);  // 5 * 0.125 = 0.625
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 30.625f);  // 0.625 + 30.0

  // Remove HEART to go below its cap
  agent->update_inventory(TestItems::HEART, -15);  // 25 * 1.0 = 25.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 25);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 25.625f);  // 0.625 + 25.0
}

// Test shared inventory limits between multiple resources
TEST_F(MettaGridCppTest, SharedInventoryLimits) {
  // Create an inventory config where ORE and LASER share a combined limit
  InventoryConfig inventory_config;
  inventory_config.limits = {
      {{TestItems::ORE, TestItems::LASER}, 30},  // ORE and LASER share a limit of 30 total
      {{TestItems::ARMOR}, 50},                  // ARMOR has its own separate limit
      {{TestItems::HEART}, 50},                  // HEART has its own separate limit
  };

  auto rewards = create_test_stats_rewards();
  auto stats_reward_max = create_test_stats_reward_max();

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0.0f, inventory_config, rewards, stats_reward_max, 0.0f, {});

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add ORE up to 20
  int delta = agent->update_inventory(TestItems::ORE, 20);
  EXPECT_EQ(delta, 20);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 20);

  // Try to add 20 LASER - should only add 10 due to shared limit
  delta = agent->update_inventory(TestItems::LASER, 20);
  EXPECT_EQ(delta, 10);  // Only 10 can be added (20 ORE + 10 LASER = 30 total)
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 10);

  // Try to add more ORE - should fail as we're at the shared limit
  delta = agent->update_inventory(TestItems::ORE, 5);
  EXPECT_EQ(delta, 0);  // Can't add any more
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 20);

  // Remove some LASER
  delta = agent->update_inventory(TestItems::LASER, -5);
  EXPECT_EQ(delta, -5);
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 5);

  // Now we can add more ORE since we freed up shared space
  delta = agent->update_inventory(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 25);

  // ARMOR should work independently with its own limit
  delta = agent->update_inventory(TestItems::ARMOR, 40);
  EXPECT_EQ(delta, 40);
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 40);

  // Can still add more ARMOR up to its limit
  delta = agent->update_inventory(TestItems::ARMOR, 20);
  EXPECT_EQ(delta, 10);  // Should cap at 50
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 50);

  // Remove all ORE
  delta = agent->update_inventory(TestItems::ORE, -25);
  EXPECT_EQ(delta, -25);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);

  // Now we can add up to 25 more LASER (5 existing + 25 = 30)
  delta = agent->update_inventory(TestItems::LASER, 30);
  EXPECT_EQ(delta, 25);
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 30);

  // Verify final state
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 30);
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 50);
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 0);
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
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(2, 3, agent_cfg, &resource_names);

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
  AgentConfig target_cfg = create_test_agent_config();
  target_cfg.group_name = "blue";
  target_cfg.group_id = 2;
  auto resource_names = create_test_resource_names();
  Agent* attacker = new Agent(2, 0, attacker_cfg, &resource_names);
  Agent* target = new Agent(0, 0, target_cfg, &resource_names);

  float attacker_reward = 0.0f;
  float target_reward = 0.0f;
  attacker->init(&attacker_reward);
  target->init(&target_reward);

  grid.add_object(attacker);
  grid.add_object(target);

  // Give attacker a laser
  attacker->update_inventory(TestItems::LASER, 2);
  EXPECT_EQ(attacker->inventory.amount(TestItems::LASER), 2);

  // Give target some items and armor
  target->update_inventory(TestItems::ARMOR, 5);
  target->update_inventory(TestItems::HEART, 3);
  EXPECT_EQ(target->inventory.amount(TestItems::ARMOR), 5);
  EXPECT_EQ(target->inventory.amount(TestItems::HEART), 3);

  // Verify attacker orientation
  EXPECT_EQ(attacker->orientation, Orientation::North);

  // Create attack action handler
  AttackActionConfig attack_cfg({{TestItems::LASER, 1}}, {{TestItems::LASER, 1}}, {{TestItems::ARMOR, 3}});
  Attack attack(attack_cfg, &game_config);
  std::mt19937 rng(42);
  attack.init(&grid, &rng);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(*attacker, 5);
  // Hitting a target with armor counts as success
  EXPECT_TRUE(success);

  // Verify that the combat material was consumed
  EXPECT_EQ(attacker->inventory.amount(TestItems::LASER), 1);
  EXPECT_EQ(target->inventory.amount(TestItems::ARMOR), 2);

  // Verify target was not frozen or robbed
  EXPECT_EQ(target->frozen, 0);
  EXPECT_EQ(target->inventory.amount(TestItems::HEART), 3);

  // Attack again, now that armor is gone
  success = attack.handle_action(*attacker, 5);
  EXPECT_TRUE(success);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory.amount(TestItems::HEART), 0);
  EXPECT_EQ(attacker->inventory.amount(TestItems::HEART), 3);
  // Humorously, the defender's armor was also stolen!
  EXPECT_EQ(target->inventory.amount(TestItems::ARMOR), 0);
  EXPECT_EQ(attacker->inventory.amount(TestItems::ARMOR), 2);
}

TEST_F(MettaGridCppTest, PutRecipeItems) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.group_name = "red";
  agent_cfg.group_id = 1;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 0, agent_cfg, &resource_names);
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
                                {10},                     // cooldown
                                0,                        // initial_resource_count
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
  bool success = put.handle_action(*agent, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);      // Ore consumed
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 1);    // Heart unchanged
  EXPECT_EQ(generator->inventory.amount(TestItems::ORE), 1);  // Ore added to generator

  // Test putting non-matching items
  success = put.handle_action(*agent, 0);
  EXPECT_FALSE(success);                                        // Should fail since we only have heart left
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 1);      // Heart unchanged
  EXPECT_EQ(generator->inventory.amount(TestItems::HEART), 0);  // No heart in generator
}

TEST_F(MettaGridCppTest, GetOutput) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.group_name = "red";
  agent_cfg.group_id = 1;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 0, agent_cfg, &resource_names);
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
                                {10},                     // cooldown
                                1,                        // initial_items
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
  bool success = get.handle_action(*agent, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 1);        // Still have ore
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 1);      // Also have armor
  EXPECT_EQ(generator->inventory.amount(TestItems::ARMOR), 0);  // Generator gave away its armor
}

// ==================== Action Tracking ====================

TEST_F(MettaGridCppTest, ActionTracking) {
  Grid grid(10, 10);

  AgentConfig agent_cfg = create_test_agent_config();
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(5, 5, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  ActionConfig noop_cfg({}, {});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 0.0f);
  noop.handle_action(*agent, 0);  // count 1, max 1
  EXPECT_EQ(agent->location.r, 5);
  EXPECT_EQ(agent->location.c, 5);
  EXPECT_EQ(agent->prev_location.r, 5);
  EXPECT_EQ(agent->prev_location.c, 5);
  EXPECT_EQ(agent->prev_action_name, "noop");
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 1.0f);
  agent->location.r = 6;
  agent->location.c = 6;
  noop.handle_action(*agent, 0);  // count 0, max 1
  EXPECT_EQ(agent->location.r, 6);
  EXPECT_EQ(agent->location.c, 6);
  EXPECT_EQ(agent->prev_location.r, 6);
  EXPECT_EQ(agent->prev_location.c, 6);
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 1.0f);
  noop.handle_action(*agent, 0);  // count 1, max 1
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 1.0f);
  noop.handle_action(*agent, 0);  // count 2, max 2
  noop.handle_action(*agent, 0);  // count 3, max 3
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 3.0f);
  agent->location.r = 7;
  agent->location.c = 7;
  noop.handle_action(*agent, 0);  // count 0, max 3
  EXPECT_EQ(agent->location.r, 7);
  EXPECT_EQ(agent->location.c, 7);
  EXPECT_EQ(agent->prev_location.r, 7);
  EXPECT_EQ(agent->prev_location.c, 7);
  noop.handle_action(*agent, 0);  // count 1, max 3
  noop.handle_action(*agent, 0);  // count 2, max 3
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 3.0f);
  noop.handle_action(*agent, 0);  // count 3, max 3
  noop.handle_action(*agent, 0);  // count 4, max 4
  EXPECT_FLOAT_EQ(agent->stats.get("status.max_steps_without_motion"), 4.0f);
}

// ==================== Fractional Consumption Tests ====================

TEST_F(MettaGridCppTest, FractionalConsumptionProbability) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
    noop.handle_action(*agent, 0);
  }

  // With 0.5 probability, exactly 4 ore should be consumed (10 - 4 = 6 remaining)
  int final_ore = agent->inventory.amount(TestItems::ORE);
  EXPECT_EQ(final_ore, 6);

  // Test that action fails when inventory is empty
  AgentConfig poor_cfg = create_test_agent_config();
  // Don't set initial_inventory so the agent starts with nothing
  Agent* poor_agent = new Agent(2, 1, poor_cfg, &resource_names);
  float poor_reward = 0.0f;
  poor_agent->init(&poor_reward);
  grid.add_object(poor_agent);

  bool success = noop.handle_action(*poor_agent, 0);
  EXPECT_FALSE(success);  // Should fail due to insufficient resources
}

TEST_F(MettaGridCppTest, FractionalConsumptionWithOverflow) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 5;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with fractional consumption (1.5)
  // Required resources must be at least ceil(consumed) = 2
  ActionConfig noop_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.5f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  bool success = noop.handle_action(*agent, 0);
  EXPECT_TRUE(success);  // Should succeed as we have enough resources

  // With 1.5, should consume either 1 or 2 units
  int final_ore = agent->inventory.amount(TestItems::ORE);
  EXPECT_TRUE(final_ore == 3 || final_ore == 4);
}

TEST_F(MettaGridCppTest, FractionalConsumptionRequiresCeiledInventory) {
  Grid grid(3, 3);

  // Create agent with only 1 resource
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 1;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create noop action with fractional consumption (1.5) - requires ceil(1.5) = 2
  ActionConfig noop_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.5f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  bool success = noop.handle_action(*agent, 0);
  EXPECT_FALSE(success);  // Should fail as we only have 1 but need ceil(1.5) = 2

  // Verify inventory unchanged
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 1);
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
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
    bool success = noop.handle_action(*agent, 0);
    EXPECT_TRUE(success);
  }

  // Verify no resources consumed
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, FractionalConsumptionInteger) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
    bool success = noop.handle_action(*agent, 0);
    EXPECT_TRUE(success);
  }

  // Verify exactly 6 resources consumed (3 * 2)
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 4);
}

TEST_F(MettaGridCppTest, FractionalConsumptionSmallFraction) {
  Grid grid(3, 3);

  // Create agent with initial energy
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 20;  // Enough for test
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
    int before = agent->inventory.amount(TestItems::ORE);
    bool success = noop.handle_action(*agent, 0);
    if (success) {
      successful_actions++;
      int after = agent->inventory.amount(TestItems::ORE);
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
  agent_cfg.initial_inventory[TestItems::ORE] = 50;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
  for (int i = 0; i < 100; i++) {
    int before = agent->inventory.amount(TestItems::ORE);
    bool success = noop.handle_action(*agent, 0);
    if (success) {
      successful_actions++;
    }
    int after = agent->inventory.amount(TestItems::ORE);
    consumed += (before - after);
  }

  EXPECT_EQ(successful_actions, 55);  // Exactly 55 successful actions before running out
  EXPECT_EQ(consumed, 50);            // All 50 ore consumed
}

TEST_F(MettaGridCppTest, FractionalConsumptionMultipleResources) {
  Grid grid(3, 3);

  // Create agent with multiple resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 50;
  agent_cfg.initial_inventory[TestItems::LASER] = 50;
  agent_cfg.initial_inventory[TestItems::ARMOR] = 50;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
    bool success = noop.handle_action(*agent, 0);
    EXPECT_TRUE(success);
  }

  int ore_left = agent->inventory.amount(TestItems::ORE);
  int laser_left = agent->inventory.amount(TestItems::LASER);
  int armor_left = agent->inventory.amount(TestItems::ARMOR);

  // Since it's random, it's okay if these values change during a refactor, as long as they stay reasonable.
  EXPECT_EQ(ore_left, 35);  // on average expect 35

  EXPECT_EQ(laser_left, 48);  // on average expect 47.5

  EXPECT_EQ(armor_left, 21);  // on average expect 22.5
}

TEST_F(MettaGridCppTest, FractionalConsumptionAttackAction) {
  // This test verifies that fractional consumption works with attack actions
  // We'll do a simple test with a few attacks rather than a complex loop

  Grid grid(10, 10);
  GameConfig game_config;

  // Create attacker with lasers
  AgentConfig attacker_cfg = create_test_agent_config();
  attacker_cfg.group_name = "red";

  // Create target
  AgentConfig target_cfg = create_test_agent_config();
  target_cfg.group_name = "blue";
  target_cfg.group_id = 2;

  auto resource_names = create_test_resource_names();
  Agent* attacker = new Agent(2, 0, attacker_cfg, &resource_names);
  Agent* target = new Agent(0, 0, target_cfg, &resource_names);

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
    int before = attacker->inventory.amount(TestItems::LASER);
    bool success = attack.handle_action(*attacker, 5);  // Attack directly in front
    if (success) {
      successful_attacks++;
      int after = attacker->inventory.amount(TestItems::LASER);
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
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
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
  while (agent->inventory.amount(TestItems::ORE) >= 2) {
    bool success = change_glyph.handle_action(*agent, (initial_glyph + 1) % 4);
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
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Test with 0.99 consumption (almost always consumes 1)
  ActionConfig noop_cfg({{TestItems::ORE, 1}}, {{TestItems::ORE, 0.99f}});
  Noop noop(noop_cfg);
  std::mt19937 rng(42);
  noop.init(&grid, &rng);

  // Should succeed once then likely fail
  bool first_success = noop.handle_action(*agent, 0);
  EXPECT_TRUE(first_success);

  // Very high chance we consumed the resource (99%)
  bool second_success = noop.handle_action(*agent, 0);
  // This will almost certainly fail (99% chance we're out of resources)
  if (agent->inventory.amount(TestItems::ORE) == 0) {
    EXPECT_FALSE(second_success);
  }
}

TEST_F(MettaGridCppTest, FractionalConsumptionDeterministicWithSameSeed) {
  Grid grid1(3, 3);
  Grid grid2(3, 3);

  // Create two identical setups
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 100;

  auto resource_names1 = create_test_resource_names();
  Agent* agent1 = new Agent(1, 1, agent_cfg, &resource_names1);
  float reward1 = 0.0f;
  agent1->init(&reward1);
  grid1.add_object(agent1);

  auto resource_names2 = create_test_resource_names();
  Agent* agent2 = new Agent(1, 1, agent_cfg, &resource_names2);
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
    noop1.handle_action(*agent1, 0);
    noop2.handle_action(*agent2, 0);
  }

  // Should have identical results with same seed
  EXPECT_EQ(agent1->inventory.amount(TestItems::ORE), agent2->inventory.amount(TestItems::ORE));
}

// ==================== Event System Tests ====================

TEST_F(MettaGridCppTest, EventManager) {
  Grid grid(10, 10);
  EventManager event_manager;

  // Test that event manager can be initialized
  // (This is a basic test - more complex event testing would require more setup)
  EXPECT_NO_THROW(event_manager.process_events(1));
}

// ==================== Assembler Tests ====================

TEST_F(MettaGridCppTest, AssemblerBasicObservationFeatures) {
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  auto features = assembler.obs_features();

  // Should have at least TypeId and Tag features
  EXPECT_GE(features.size(), 3);  // TypeId + 2 tags

  // Find TypeId feature
  bool found_type_id = false;
  bool found_tag1 = false;
  bool found_tag2 = false;

  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::TypeId) {
      EXPECT_EQ(feature.value, 1);  // Our test assembler type_id
      found_type_id = true;
    } else if (feature.feature_id == ObservationFeature::Tag) {
      if (feature.value == 1) {
        found_tag1 = true;
      } else if (feature.value == 2) {
        found_tag2 = true;
      }
    }
  }

  EXPECT_TRUE(found_type_id) << "TypeId feature not found";
  EXPECT_TRUE(found_tag1) << "Tag 1 not found";
  EXPECT_TRUE(found_tag2) << "Tag 2 not found";
}

TEST_F(MettaGridCppTest, AssemblerNoCooldownObservation) {
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  // Initially no cooldown
  auto features = assembler.obs_features();

  // Should not have CooldownRemaining feature when not cooling down
  bool found_cooldown_remaining = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::CooldownRemaining) {
      found_cooldown_remaining = true;
      break;
    }
  }
  EXPECT_FALSE(found_cooldown_remaining) << "Should not have CooldownRemaining feature when not cooling down";
}

TEST_F(MettaGridCppTest, AssemblerCooldownRemainingCalculation) {
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  // Test cooldown_remaining() function directly

  // Initially no cooldown
  EXPECT_EQ(assembler.cooldown_remaining(), 0);

  // Set cooldown end timestep
  assembler.cooldown_end_timestep = 10;
  current_timestep = 5;

  // Should have 5 remaining
  EXPECT_EQ(assembler.cooldown_remaining(), 5);

  // Advance time
  current_timestep = 8;
  EXPECT_EQ(assembler.cooldown_remaining(), 2);

  // At end time
  current_timestep = 10;
  EXPECT_EQ(assembler.cooldown_remaining(), 0);

  // Past end time
  current_timestep = 15;
  EXPECT_EQ(assembler.cooldown_remaining(), 0);
}

TEST_F(MettaGridCppTest, AssemblerCooldownObservationWithRemainingTime) {
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  // Set up cooldown
  assembler.cooldown_end_timestep = 10;
  current_timestep = 5;

  auto features = assembler.obs_features();

  // Should have CooldownRemaining feature
  bool found_cooldown_remaining = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::CooldownRemaining) {
      EXPECT_EQ(feature.value, 5);  // 10 - 5 = 5 remaining
      found_cooldown_remaining = true;
      break;
    }
  }
  EXPECT_TRUE(found_cooldown_remaining) << "Should have CooldownRemaining feature when cooling down";
}

TEST_F(MettaGridCppTest, AssemblerCooldownObservationCappedAt255) {
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  // Set up a very long cooldown
  assembler.cooldown_end_timestep = 1000;
  current_timestep = 100;  // 900 remaining, but should be capped at 255

  auto features = assembler.obs_features();

  bool found_cooldown_remaining = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::CooldownRemaining) {
      EXPECT_EQ(feature.value, 255);  // Should be capped at 255
      found_cooldown_remaining = true;
      break;
    }
  }
  EXPECT_TRUE(found_cooldown_remaining) << "Should have CooldownRemaining feature capped at 255";
}

TEST_F(MettaGridCppTest, AssemblerGetAgentPatternByte) {
  // Create a grid to test with
  std::unique_ptr<Grid> grid = std::make_unique<Grid>(10, 10);

  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  Assembler* assembler = new Assembler(5, 5, config);  // Assembler at position (5,5)

  // Set up the assembler with grid
  assembler->set_grid(grid.get());
  grid->add_object(assembler);

  // Test 1: Empty pattern (no agents around) - should return 0
  uint8_t pattern = assembler->get_agent_pattern_byte();

  AgentConfig agent_cfg(1, "test_agent", 0, "test_group");
  auto resource_names = create_test_resource_names();
  Agent* agent1 = new Agent(4, 5, agent_cfg, &resource_names);  // North of assembler
  Agent* agent2 = new Agent(5, 6, agent_cfg, &resource_names);  // East of assembler

  grid->add_object(agent1);
  grid->add_object(agent2);

  pattern = assembler->get_agent_pattern_byte();
  EXPECT_EQ(pattern, 18) << "Pattern with agents at N and E should be 18 (2 + 16)";

  // Test 3: Pattern with agents in multiple positions
  // Move agent1 to NW (bit 0) and agent2 to SW (bit 5), add agent3 at SE (bit 7)
  // This should give us pattern = (1 << 0) | (1 << 5) | (1 << 7) = 1 | 32 | 128 = 161
  grid->move_object(*agent1, GridLocation(4, 4, GridLayer::AgentLayer));  // Move to NW
  grid->move_object(*agent2, GridLocation(6, 4, GridLayer::AgentLayer));  // Move to SW

  Agent* agent3 = new Agent(6, 6, agent_cfg, &resource_names);  // SE of assembler
  grid->add_object(agent3);                                     // Add new agent

  pattern = assembler->get_agent_pattern_byte();
  EXPECT_EQ(pattern, 161) << "Pattern with agents at NW, SW, and SE should be 161 (1 + 32 + 128)";
}

TEST_F(MettaGridCppTest, AssemblerGetCurrentRecipe) {
  // Create a grid to test with
  Grid grid(10, 10);

  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});

  // Create test recipes
  auto recipe0 = std::make_shared<Recipe>();
  recipe0->input_resources[0] = 1;

  auto recipe1 = std::make_shared<Recipe>();
  recipe1->input_resources[1] = 2;

  config.recipes.push_back(recipe0);
  config.recipes.push_back(recipe1);

  Assembler* assembler = new Assembler(5, 5, config);

  // Set up the assembler with grid and timestep
  unsigned int current_timestep = 0;
  assembler->set_current_timestep_ptr(&current_timestep);
  assembler->set_grid(&grid);

  // Add assembler to grid
  grid.add_object(assembler);

  // Without agents around, should get pattern 0 (recipe0)
  const Recipe* current_recipe = assembler->get_current_recipe();
  EXPECT_EQ(current_recipe, recipe0.get());

  // Add one agent at NW position (bit 0) - should get pattern 1 (recipe1)
  AgentConfig agent_cfg(1, "test_agent", 0, "test_group");
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(4, 4, agent_cfg, &resource_names);  // NW of assembler
  grid.add_object(agent);

  current_recipe = assembler->get_current_recipe();
  EXPECT_EQ(current_recipe, recipe1.get()) << "With one agent, should select recipe1";
}

TEST_F(MettaGridCppTest, AssemblerRecipeObservationsEnabled) {
  // Create a grid to test with
  Grid grid(10, 10);

  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  config.recipe_details_obs = true;
  config.input_recipe_offset = 100;
  config.output_recipe_offset = 200;

  // Create test recipes - one for pattern 0 (no agents), one for pattern 1 (some agents)
  auto recipe0 = std::make_shared<Recipe>();
  recipe0->input_resources[0] = 2;   // 2 units of item 0
  recipe0->output_resources[1] = 1;  // 1 unit of output item 1

  auto recipe1 = std::make_shared<Recipe>();
  recipe1->input_resources[2] = 3;   // 3 units of item 2
  recipe1->output_resources[3] = 2;  // 2 units of output item 3

  config.recipes.push_back(recipe0);  // Index 0: pattern 0
  config.recipes.push_back(recipe1);  // Index 1: pattern 1

  Assembler* assembler = new Assembler(5, 5, config);

  // Set up the assembler with grid and timestep
  unsigned int current_timestep = 0;
  assembler->set_current_timestep_ptr(&current_timestep);
  assembler->set_grid(&grid);

  // Add assembler to grid
  grid.add_object(assembler);

  // Test with pattern 0 (no agents around) - should get recipe0
  auto features = assembler->obs_features();

  // Should have recipe features for pattern 0 (recipe0)
  bool found_input_feature = false;
  bool found_output_feature = false;
  for (const auto& feature : features) {
    if (feature.feature_id == config.input_recipe_offset + 0) {
      EXPECT_EQ(feature.value, 2);  // 2 units of input item 0 from recipe0
      found_input_feature = true;
    } else if (feature.feature_id == config.output_recipe_offset + 1) {
      EXPECT_EQ(feature.value, 1);  // 1 unit of output item 1 from recipe0
      found_output_feature = true;
    }
  }
  EXPECT_TRUE(found_input_feature) << "Should have input recipe feature for pattern 0";
  EXPECT_TRUE(found_output_feature) << "Should have output recipe feature for pattern 0";

  // Verify we're getting the right recipe
  const Recipe* current_recipe = assembler->get_current_recipe();
  EXPECT_EQ(current_recipe, recipe0.get());
}

TEST_F(MettaGridCppTest, AssemblerBalancedConsumptionAmpleResources) {
  // Test case (a): 3 agents with ample resources, consume 10 total
  // Each agent should lose 3-4 resources for balanced consumption

  // Create a recipe that requires 10 ore
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  input_resources[TestItems::ORE] = 10;

  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  output_resources[TestItems::LASER] = 1;

  auto recipe = std::make_shared<Recipe>(input_resources, output_resources, 0);

  // Create assembler with the recipe
  AssemblerConfig config(1, "test_assembler", std::vector<int>{});
  config.recipes = {recipe};
  Assembler assembler(5, 5, config);

  // Create agents with ample resources
  AgentConfig agent_config(0, "agent", 0, "agent");
  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_config, &resource_names);
  Agent agent2(0, 0, agent_config, &resource_names);
  Agent agent3(0, 0, agent_config, &resource_names);

  agent1.update_inventory(TestItems::ORE, 20);
  agent2.update_inventory(TestItems::ORE, 20);
  agent3.update_inventory(TestItems::ORE, 20);

  std::vector<Agent*> surrounding_agents = {&agent1, &agent2, &agent3};

  // Consume resources
  assembler.consume_resources_for_recipe(*recipe, surrounding_agents);

  // Check balanced consumption
  InventoryQuantity consumed1 = 20 - agent1.inventory.amount(TestItems::ORE);
  InventoryQuantity consumed2 = 20 - agent2.inventory.amount(TestItems::ORE);
  InventoryQuantity consumed3 = 20 - agent3.inventory.amount(TestItems::ORE);

  // Total should be exactly 10
  EXPECT_EQ(consumed1 + consumed2 + consumed3, 10);

  // Each agent should lose 3-4 resources (balanced)
  // With 10 resources and 3 agents: 10/3 = 3.33, so we expect 3, 3, 4 distribution
  EXPECT_GE(consumed1, 3);
  EXPECT_LE(consumed1, 4);
  EXPECT_GE(consumed2, 3);
  EXPECT_LE(consumed2, 4);
  EXPECT_GE(consumed3, 3);
  EXPECT_LE(consumed3, 4);
}

TEST_F(MettaGridCppTest, AssemblerBalancedConsumptionMixedResources) {
  // Test case (b): 4 agents with mixed resources
  // Agent 1: 0 resources, Agent 2: 1 resource, Agents 3&4: ample resources
  // When consuming 20, should consume 0/1/9/10 respectively

  // Create a recipe that requires 20 ore
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  input_resources[TestItems::ORE] = 20;

  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  output_resources[TestItems::LASER] = 1;

  auto recipe = std::make_shared<Recipe>(input_resources, output_resources, 0);

  // Create assembler with the recipe
  AssemblerConfig config(1, "test_assembler", std::vector<int>{});
  config.recipes = {recipe};
  Assembler assembler(5, 5, config);

  // Create agents with varied resources
  AgentConfig agent_config(0, "agent", 0, "agent");
  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_config, &resource_names);
  Agent agent2(0, 0, agent_config, &resource_names);
  Agent agent3(0, 0, agent_config, &resource_names);
  Agent agent4(0, 0, agent_config, &resource_names);

  agent1.update_inventory(TestItems::ORE, 0);   // No resources
  agent2.update_inventory(TestItems::ORE, 1);   // Limited resources
  agent3.update_inventory(TestItems::ORE, 50);  // Ample resources
  agent4.update_inventory(TestItems::ORE, 50);  // Ample resources

  std::vector<Agent*> surrounding_agents = {&agent1, &agent2, &agent3, &agent4};

  // Consume resources
  assembler.consume_resources_for_recipe(*recipe, surrounding_agents);

  // Check consumption matches expected pattern
  InventoryQuantity consumed1 = 0 - agent1.inventory.amount(TestItems::ORE);
  InventoryQuantity consumed2 = 1 - agent2.inventory.amount(TestItems::ORE);
  InventoryQuantity consumed3 = 50 - agent3.inventory.amount(TestItems::ORE);
  InventoryQuantity consumed4 = 50 - agent4.inventory.amount(TestItems::ORE);

  // Total should be exactly 20
  EXPECT_EQ(consumed1 + consumed2 + consumed3 + consumed4, 20);

  // Expected consumption pattern: 0, 1, 9-10, 9-10
  EXPECT_EQ(consumed1, 0) << "Agent with 0 resources should consume 0";
  EXPECT_EQ(consumed2, 1) << "Agent with 1 resource should consume 1";

  // Remaining 19 should be split between agents 3 and 4 (9 and 10 or 10 and 9)
  EXPECT_GE(consumed3, 9);
  EXPECT_LE(consumed3, 10);
  EXPECT_GE(consumed4, 9);
  EXPECT_LE(consumed4, 10);
  EXPECT_EQ(consumed3 + consumed4, 19) << "Agents 3 and 4 should consume the remaining 19";
}

TEST_F(MettaGridCppTest, AssemblerClippingAndUnclipping) {
  // Create a simple grid
  Grid grid(10, 10);
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  unsigned int current_timestep = 0;

  // Create an assembler with normal recipes
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});

  // Create normal recipes (pattern 0: no agents needed)
  auto normal_recipe = std::make_shared<Recipe>();
  normal_recipe->input_resources[TestItems::ORE] = 2;
  normal_recipe->output_resources[TestItems::LASER] = 1;
  normal_recipe->cooldown = 0;

  config.recipes.resize(256);
  for (int i = 0; i < 256; i++) {
    config.recipes[i] = normal_recipe;
  }

  Assembler assembler(5, 5, config);
  assembler.set_grid(&grid);
  assembler.set_current_timestep_ptr(&current_timestep);

  // Create an agent to interact with the assembler
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;
  agent_cfg.initial_inventory[TestItems::HEART] = 5;

  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(4, 5, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->reward = &agent_reward;
  grid.add_object(agent);

  // Test 1: Verify assembler is not clipped initially
  EXPECT_FALSE(assembler.is_clipped) << "Assembler should not be clipped initially";

  // Test 2: Verify normal recipe works when not clipped
  bool success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use normal recipe when not clipped";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 8) << "Should consume 2 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 1) << "Should produce 1 laser";

  // Test 3: Create unclipping recipes and clip the assembler
  auto unclip_recipe = std::make_shared<Recipe>();
  unclip_recipe->input_resources[TestItems::HEART] = 1;
  unclip_recipe->output_resources[TestItems::ORE] = 3;
  unclip_recipe->cooldown = 0;

  std::vector<std::shared_ptr<Recipe>> unclip_recipes(256, unclip_recipe);
  assembler.become_clipped(unclip_recipes, nullptr);

  EXPECT_TRUE(assembler.is_clipped) << "Assembler should be clipped after become_clipped()";
  EXPECT_EQ(assembler.unclip_recipes.size(), 256) << "Should have unclip recipes set";

  // Test 4: Verify clipped observation feature
  auto features = assembler.obs_features();
  bool found_clipped = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::Clipped) {
      EXPECT_EQ(feature.value, 1) << "Clipped observation should be 1";
      found_clipped = true;
      break;
    }
  }
  EXPECT_TRUE(found_clipped) << "Should have Clipped observation feature when clipped";

  // Test 5: Verify unclip recipe is used when clipped
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use unclip recipe when clipped";
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 4) << "Should consume 1 heart for unclipping";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 11) << "Should produce 3 ore from unclip recipe";

  // Test 6: Verify assembler is automatically unclipped after successful use
  EXPECT_FALSE(assembler.is_clipped) << "Assembler should be unclipped after successful use";
  EXPECT_TRUE(assembler.unclip_recipes.empty()) << "Unclip recipes should be cleared";

  // Test 7: Verify normal recipe works again after unclipping
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use normal recipe after unclipping";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 9) << "Should consume 2 ore (normal recipe)";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 2) << "Should produce 1 more laser";

  // Test 8: Verify no clipped observation after unclipping
  features = assembler.obs_features();
  found_clipped = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::Clipped) {
      found_clipped = true;
      break;
    }
  }
  EXPECT_FALSE(found_clipped) << "Should not have Clipped observation feature when not clipped";
}

TEST_F(MettaGridCppTest, AssemblerMaxUses) {
  // Create a simple grid
  Grid grid(10, 10);
  unsigned int current_timestep = 0;

  // Create an assembler with max_uses set to 3
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  config.max_uses = 3;  // Limit to 3 uses

  // Create simple recipe
  auto recipe = std::make_shared<Recipe>();
  recipe->input_resources[TestItems::ORE] = 1;
  recipe->output_resources[TestItems::LASER] = 1;
  recipe->cooldown = 0;

  config.recipes.resize(256);
  for (int i = 0; i < 256; i++) {
    config.recipes[i] = recipe;
  }

  Assembler assembler(5, 5, config);
  assembler.set_grid(&grid);
  assembler.set_current_timestep_ptr(&current_timestep);

  // Create an agent with plenty of resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;

  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(4, 5, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->reward = &agent_reward;
  grid.add_object(agent);

  // Test 1: Verify initial state
  EXPECT_EQ(assembler.max_uses, 3) << "Max uses should be 3";
  EXPECT_EQ(assembler.uses_count, 0) << "Uses count should be 0 initially";

  // Test 2: Verify remaining uses in observations
  auto features = assembler.obs_features();
  bool found_remaining_uses = false;
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::RemainingUses) {
      EXPECT_EQ(feature.value, 3) << "Should show 3 remaining uses";
      found_remaining_uses = true;
      break;
    }
  }
  EXPECT_TRUE(found_remaining_uses) << "Should have RemainingUses observation when max_uses is set";

  // Test 3: First use should succeed
  bool success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "First use should succeed";
  EXPECT_EQ(assembler.uses_count, 1) << "Uses count should be 1";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 9) << "Should consume 1 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 1) << "Should produce 1 laser";

  // Test 4: Check remaining uses after first use
  features = assembler.obs_features();
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::RemainingUses) {
      EXPECT_EQ(feature.value, 2) << "Should show 2 remaining uses";
      break;
    }
  }

  // Test 5: Second use should succeed
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Second use should succeed";
  EXPECT_EQ(assembler.uses_count, 2) << "Uses count should be 2";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 2) << "Should have 2 lasers";

  // Test 6: Third use should succeed
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Third use should succeed";
  EXPECT_EQ(assembler.uses_count, 3) << "Uses count should be 3";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 3) << "Should have 3 lasers";

  // Test 7: Check remaining uses after max reached
  features = assembler.obs_features();
  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::RemainingUses) {
      EXPECT_EQ(feature.value, 0) << "Should show 0 remaining uses";
      break;
    }
  }

  // Test 8: Fourth use should fail (max uses reached)
  success = assembler.onUse(*agent, 0);
  EXPECT_FALSE(success) << "Fourth use should fail - max uses reached";
  EXPECT_EQ(assembler.uses_count, 3) << "Uses count should still be 3";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 7) << "Should still have 7 ore (no consumption)";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 3) << "Should still have 3 lasers (no production)";
}

TEST_F(MettaGridCppTest, AssemblerExhaustion) {
  // Create a simple grid
  Grid grid(10, 10);
  unsigned int current_timestep = 0;

  // Create an assembler with exhaustion enabled
  AssemblerConfig config(1, "test_assembler", std::vector<int>{1, 2});
  config.exhaustion = 0.5f;  // 50% exhaustion rate - multiplier grows by 1.5x each use

  // Create recipe with cooldown
  auto recipe = std::make_shared<Recipe>();
  recipe->input_resources[TestItems::ORE] = 1;
  recipe->output_resources[TestItems::LASER] = 1;
  recipe->cooldown = 10;  // Base cooldown of 10 timesteps

  config.recipes.resize(256);
  for (int i = 0; i < 256; i++) {
    config.recipes[i] = recipe;
  }

  Assembler assembler(5, 5, config);
  assembler.set_grid(&grid);
  assembler.set_current_timestep_ptr(&current_timestep);

  // Create an agent with plenty of resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;

  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(4, 5, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->reward = &agent_reward;
  grid.add_object(agent);

  // Test 1: Verify initial state
  EXPECT_EQ(assembler.exhaustion, 0.5f) << "Exhaustion rate should be 0.5";
  EXPECT_EQ(assembler.cooldown_multiplier, 1.0f) << "Initial cooldown multiplier should be 1.0";

  // Test 2: First use should have normal cooldown
  bool success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "First use should succeed";
  EXPECT_EQ(assembler.cooldown_end_timestep, 10) << "First cooldown should be 10 (base cooldown)";
  EXPECT_EQ(assembler.cooldown_multiplier, 1.5f) << "Cooldown multiplier should be 1.5 after first use";

  // Test 3: Wait for cooldown and use again
  current_timestep = 10;

  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Second use should succeed";
  // Second cooldown should be 10 * 1.5 = 15
  EXPECT_EQ(assembler.cooldown_end_timestep, 25) << "Second cooldown should end at 25 (10 + 15)";
  EXPECT_FLOAT_EQ(assembler.cooldown_multiplier, 2.25f) << "Cooldown multiplier should be 2.25 after second use";

  // Test 4: Third use should have even longer cooldown
  current_timestep = 25;
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Third use should succeed";
  // Third cooldown should be 10 * 2.25 = 22.5, rounded to 22
  EXPECT_EQ(assembler.cooldown_end_timestep, 47) << "Third cooldown should end at 47 (25 + 22)";
  EXPECT_FLOAT_EQ(assembler.cooldown_multiplier, 3.375f) << "Cooldown multiplier should be 3.375 after third use";

  // Test 5: Verify exhaustion grows exponentially
  current_timestep = 47;
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Fourth use should succeed";
  // Fourth cooldown should be 10 * 3.375 = 33.75, rounded to 33
  EXPECT_EQ(assembler.cooldown_end_timestep, 80) << "Fourth cooldown should end at 80 (47 + 33)";
  EXPECT_FLOAT_EQ(assembler.cooldown_multiplier, 5.0625f) << "Cooldown multiplier should be 5.0625 after fourth use";
}

// ==================== ResourceMod Tests ====================

TEST_F(MettaGridCppTest, ResourceModBasic) {
  Grid grid(5, 5);
  std::mt19937 rng(42);
  auto resource_names = create_test_resource_names();

  // Create actor at center
  AgentConfig actor_cfg = create_test_agent_config();
  actor_cfg.initial_inventory[TestItems::ORE] = 10;
  Agent* actor = new Agent(2, 2, actor_cfg, &resource_names);
  float actor_reward = 0.0f;
  actor->init(&actor_reward);
  grid.add_object(actor);

  // Create target agent nearby
  AgentConfig target_cfg = create_test_agent_config();
  target_cfg.initial_inventory[TestItems::HEART] = 10;
  Agent* target = new Agent(2, 3, target_cfg, &resource_names);
  float target_reward = 0.0f;
  target->init(&target_reward);
  grid.add_object(target);

  // Create resource mod action that adds hearts with 100% probability
  ResourceModConfig modify_cfg({{TestItems::ORE, 1}},       // required_resources
                               {{TestItems::ORE, 1.0f}},    // consumed_resources
                               {{TestItems::HEART, 1.0f}},  // modifies - adds 1 heart
                               1,                           // agent_radius
                               0,                           // converter_radius
                               false);                      // scales
  ResourceMod modify(modify_cfg);
  modify.init(&grid, &rng);

  ActionArg arg = 0;  // Unused
  bool success = modify.handle_action(*actor, arg);
  EXPECT_TRUE(success);

  // Check that target gained 1 heart
  EXPECT_EQ(target->inventory.amount(TestItems::HEART), 11);
  // Check that actor lost 1 ore
  EXPECT_EQ(actor->inventory.amount(TestItems::ORE), 9);
}

TEST_F(MettaGridCppTest, ResourceModProbabilistic) {
  Grid grid(5, 5);
  std::mt19937 rng(42);
  auto resource_names = create_test_resource_names();

  // Create actor
  AgentConfig actor_cfg = create_test_agent_config();
  actor_cfg.initial_inventory[TestItems::ORE] = 200;
  Agent* actor = new Agent(2, 2, actor_cfg, &resource_names);
  float actor_reward = 0.0f;
  actor->init(&actor_reward);
  grid.add_object(actor);

  // Create target
  AgentConfig target_cfg = create_test_agent_config();
  target_cfg.initial_inventory[TestItems::HEART] = 10;
  Agent* target = new Agent(2, 3, target_cfg, &resource_names);
  float target_reward = 0.0f;
  target->init(&target_reward);
  grid.add_object(target);

  // Create action with fractional modifications (30% chance)
  ResourceModConfig modify_cfg({{TestItems::ORE, 1}},       // required_resources must have ceil(0.5) = 1
                               {{TestItems::ORE, 0.5f}},    // 50% chance to consume
                               {{TestItems::HEART, 0.3f}},  // 30% chance to add 1 heart
                               1,
                               0,
                               false);  // radius 1, no converters, no scaling
  ResourceMod modify(modify_cfg);
  modify.init(&grid, &rng);

  // Execute multiple times to test probabilistic behavior
  ActionArg arg = 0;  // Unused
  int hearts_added = 0;
  int ore_consumed = 0;

  for (int i = 0; i < 100; i++) {
    int ore_before = actor->inventory.amount(TestItems::ORE);
    int hearts_before = target->inventory.amount(TestItems::HEART);

    // Check if actor has required resources
    if (ore_before < 1) {
      // Actor is out of ore, can't continue test
      break;
    }

    bool success = modify.handle_action(*actor, arg);
    EXPECT_TRUE(success);

    ore_consumed += (ore_before - actor->inventory.amount(TestItems::ORE));
    hearts_added += (target->inventory.amount(TestItems::HEART) - hearts_before);
  }

  // With 30% probability for hearts and 50% for ore consumption
  // Expect around 30 hearts added and 50 ore consumed
  EXPECT_GE(hearts_added, 20);  // At least 20
  EXPECT_LE(hearts_added, 40);  // At most 40
  EXPECT_GE(ore_consumed, 40);  // At least 40
  EXPECT_LE(ore_consumed, 60);  // At most 60
}

TEST_F(MettaGridCppTest, ResourceModConverter) {
  Grid grid(5, 5);
  std::mt19937 rng(42);
  EventManager event_manager;
  auto resource_names = create_test_resource_names();

  // Create actor
  AgentConfig actor_cfg = create_test_agent_config();
  Agent* actor = new Agent(2, 2, actor_cfg, &resource_names);
  float actor_reward = 0.0f;
  actor->init(&actor_reward);
  grid.add_object(actor);

  // Create converter nearby
  ConverterConfig converter_cfg(TestItems::CONVERTER,  // type_id
                                "converter",           // type_name
                                {},                    // input_resources
                                {},                    // output_resources
                                -1,                    // max_output
                                -1,                    // max_conversions
                                0,                     // conversion_ticks
                                {0},                   // cooldown
                                0,                     // initial_items
                                false);                // recipe_details_obs
  Converter* converter = new Converter(3, 2, converter_cfg);
  grid.add_object(converter);
  converter->set_event_manager(&event_manager);

  // Create action that modifies converter resources
  ResourceModConfig modify_cfg({},
                               {},
                               {{TestItems::ORE, 1.0f}},  // Add 1 ore to converter
                               0,
                               1,
                               false);  // No agents, converters within radius 1
  ResourceMod modify(modify_cfg);
  modify.init(&grid, &rng);

  // Target converter at (3, 2) from actor at (2, 2)
  ActionArg arg = 0;  // Unused
  bool success = modify.handle_action(*actor, arg);
  EXPECT_TRUE(success);

  // Check that converter gained 1 ore
  EXPECT_EQ(converter->inventory.amount(TestItems::ORE), 1);
}

TEST_F(MettaGridCppTest, ConverterCooldownSequenceCycles) {
  Grid grid(5, 5);
  EventManager event_manager;
  event_manager.init(&grid);
  RegisterProductionHandlers(event_manager);

  std::vector<unsigned short> cooldown_time_values{2, 4, 0};
  ConverterConfig converter_cfg(
      TestItems::CONVERTER, "converter", {}, {{TestItems::ORE, 1}}, -1, -1, 1, cooldown_time_values);
  Converter* converter = new Converter(2, 2, converter_cfg);
  grid.add_object(converter);
  converter->set_event_manager(&event_manager);

  std::vector<unsigned int> completions;
  unsigned int last_output = 0;
  const unsigned int total_steps = 40;
  for (unsigned int step = 0; step <= total_steps; ++step) {
    event_manager.process_events(step);
    unsigned short current_output = converter->inventory.amount(TestItems::ORE);
    if (current_output > last_output) {
      completions.push_back(step);
      last_output = current_output;
    }
  }

  std::vector<unsigned short> observed;
  for (size_t i = 1; i < completions.size(); ++i) {
    unsigned int gap = completions[i] - completions[i - 1];
    unsigned short cooldown = gap > 1 ? static_cast<unsigned short>(gap - 1) : 0;
    observed.push_back(cooldown);
  }

  std::vector<unsigned short> expected{2, 4, 0, 2, 4};
  ASSERT_GE(observed.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(observed[i], expected[i]);
  }

  EXPECT_EQ(converter->cooldown_time, cooldown_time_values);
}

TEST_F(MettaGridCppTest, ConverterCooldownSequenceHandlesEmptyList) {
  Grid grid(5, 5);
  EventManager event_manager;
  event_manager.init(&grid);
  RegisterProductionHandlers(event_manager);

  std::vector<unsigned short> cooldown_time_values;
  ConverterConfig converter_cfg(
      TestItems::CONVERTER, "converter", {}, {{TestItems::ORE, 1}}, -1, -1, 1, cooldown_time_values);
  Converter* converter = new Converter(1, 1, converter_cfg);
  grid.add_object(converter);
  converter->set_event_manager(&event_manager);

  std::vector<unsigned int> completions;
  unsigned int last_output = 0;
  const unsigned int total_steps = 12;
  for (unsigned int step = 0; step <= total_steps; ++step) {
    event_manager.process_events(step);
    unsigned short current_output = converter->inventory.amount(TestItems::ORE);
    if (current_output > last_output) {
      completions.push_back(step);
      last_output = current_output;
    }
  }

  std::vector<unsigned short> observed;
  for (size_t i = 1; i < completions.size(); ++i) {
    unsigned int gap = completions[i] - completions[i - 1];
    unsigned short cooldown = gap > 1 ? static_cast<unsigned short>(gap - 1) : 0;
    observed.push_back(cooldown);
  }

  std::vector<unsigned short> expected(observed.size(), 0);
  EXPECT_EQ(observed, expected);
}

TEST_F(MettaGridCppTest, ConverterRespectsMaxConversionsLimit) {
  Grid grid(5, 5);
  EventManager event_manager;
  event_manager.init(&grid);
  RegisterProductionHandlers(event_manager);

  std::vector<unsigned short> cooldown_time_values{5, 10};
  ConverterConfig converter_cfg(
      TestItems::CONVERTER, "converter", {}, {{TestItems::ORE, 1}}, -1, 2, 1, cooldown_time_values);
  Converter* converter = new Converter(3, 3, converter_cfg);
  grid.add_object(converter);
  converter->set_event_manager(&event_manager);

  std::vector<unsigned int> completions;
  unsigned int last_output = 0;
  const unsigned int total_steps = 40;
  for (unsigned int step = 0; step <= total_steps; ++step) {
    event_manager.process_events(step);
    unsigned short current_output = converter->inventory.amount(TestItems::ORE);
    if (current_output > last_output) {
      completions.push_back(step);
      last_output = current_output;
    }
  }

  EXPECT_EQ(completions.size(), 2u);
  EXPECT_EQ(converter->inventory.amount(TestItems::ORE), 2);
}

// Tests for Inventory::shared_update function
TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_EvenDistribution) {
  // Test that positive delta is evenly distributed among inventories
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Add 30 ore, should be distributed as 10 each
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 30);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv3.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_UnevenDistribution) {
  // Test that when delta doesn't divide evenly, earlier inventories get more
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Add 31 ore, should be distributed as 11, 10, 10 (earlier inventories get more)
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 31);

  EXPECT_EQ(consumed, 31);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 11);
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv3.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_WithLimits) {
  // Test that inventories that hit their limit drop out of distribution
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 10}};  // Low limit of 10

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  // Pre-fill inv1 with 5 ore
  inv1.update(TestItems::ORE, 5);

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Try to add 30 ore
  // inv1 can only take 5 more (to reach limit of 10)
  // inv2 and inv3 can each take 10 (to reach their limits)
  // Total consumed will be 5 + 10 + 10 = 25, not the full 30
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 25);                     // Only 25 can be consumed due to limits
  EXPECT_EQ(inv1.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(inv3.amount(TestItems::ORE), 10);  // Hit limit
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_EvenDistribution) {
  // Test that negative delta is evenly distributed among inventories
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  // Pre-fill inventories with 20 ore each
  inv1.update(TestItems::ORE, 20);
  inv2.update(TestItems::ORE, 20);
  inv3.update(TestItems::ORE, 20);

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Remove 30 ore, should remove 10 from each
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv3.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_InsufficientResources) {
  // Test behavior when some inventories don't have enough to contribute their share
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  // Pre-fill inventories with different amounts
  inv1.update(TestItems::ORE, 5);   // Only has 5
  inv2.update(TestItems::ORE, 20);  // Has plenty
  inv3.update(TestItems::ORE, 20);  // Has plenty

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Try to remove 30 ore
  // inv1 can only contribute 5, remaining 25 split between inv2 and inv3 as 13, 12
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 0);  // Depleted
  EXPECT_EQ(inv2.amount(TestItems::ORE), 7);  // 20 - 13
  EXPECT_EQ(inv3.amount(TestItems::ORE), 8);  // 20 - 12
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_UnevenDistribution) {
  // Test that when negative delta doesn't divide evenly, earlier inventories lose more
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);
  Inventory inv3(cfg);

  // Pre-fill inventories with 20 ore each
  inv1.update(TestItems::ORE, 20);
  inv2.update(TestItems::ORE, 20);
  inv3.update(TestItems::ORE, 20);

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Remove 31 ore, should remove 11, 10, 10 (earlier inventories lose more)
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, -31);

  EXPECT_EQ(consumed, -31);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 9);   // 20 - 11
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);  // 20 - 10
  EXPECT_EQ(inv3.amount(TestItems::ORE), 10);  // 20 - 10
}

TEST_F(MettaGridCppTest, SharedUpdate_EmptyInventoriesList) {
  // Test with empty inventories list
  std::vector<Inventory*> inventories;

  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 10);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since no inventories
}

TEST_F(MettaGridCppTest, SharedUpdate_SingleInventory) {
  // Test with single inventory
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 100}};

  Inventory inv1(cfg);
  std::vector<Inventory*> inventories = {&inv1};

  // All delta should go to the single inventory
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 25);

  EXPECT_EQ(consumed, 25);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 25);
}

TEST_F(MettaGridCppTest, SharedUpdate_AllInventoriesAtLimit) {
  // Test when all inventories are at their limit
  InventoryConfig cfg;
  cfg.limits = {{{TestItems::ORE}, 10}};

  Inventory inv1(cfg);
  Inventory inv2(cfg);

  // Fill both to limit
  inv1.update(TestItems::ORE, 10);
  inv2.update(TestItems::ORE, 10);

  std::vector<Inventory*> inventories = {&inv1, &inv2};

  // Try to add more
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 20);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since all at limit
  EXPECT_EQ(inv1.amount(TestItems::ORE), 10);
  EXPECT_EQ(inv2.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_MixedLimits) {
  // Test with inventories having different limits
  InventoryConfig cfg1;
  cfg1.limits = {{{TestItems::ORE}, 10}};

  InventoryConfig cfg2;
  cfg2.limits = {{{TestItems::ORE}, 20}};

  InventoryConfig cfg3;
  cfg3.limits = {{{TestItems::ORE}, 30}};

  Inventory inv1(cfg1);  // Limit 10
  Inventory inv2(cfg2);  // Limit 20
  Inventory inv3(cfg3);  // Limit 30

  std::vector<Inventory*> inventories = {&inv1, &inv2, &inv3};

  // Try to add 45 ore
  // inv1 takes 10 (hits limit), inv2 takes 18, inv3 takes 17
  InventoryDelta consumed = Inventory::shared_update(inventories, TestItems::ORE, 45);

  EXPECT_EQ(consumed, 45);
  EXPECT_EQ(inv1.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(inv2.amount(TestItems::ORE), 18);  // Gets more due to being earlier
  EXPECT_EQ(inv3.amount(TestItems::ORE), 17);
}
