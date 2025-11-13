#include <gtest/gtest.h>

#include <array>
#include <random>
#include <utility>

#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/noop.hpp"
#include "actions/resource_mod.hpp"
#include "config/mettagrid_config.hpp"
#include "config/observation_features.hpp"
#include "core/event.hpp"
#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/agent_config.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"
#include "objects/protocol.hpp"
#include "objects/wall.hpp"

// Test-specific inventory item type constants
namespace TestItems {
constexpr uint8_t ORE = 0;
constexpr uint8_t LASER = 1;
constexpr uint8_t ARMOR = 2;
constexpr uint8_t HEART = 3;
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
  void SetUp() override {
    // Initialize ObservationFeature constants for tests
    // Use standard feature IDs that match what the game would use
    std::unordered_map<std::string, ObservationType> feature_ids = {
        {"type_id", 0},
        {"agent:group", 1},
        {"agent:frozen", 2},
        {"agent:orientation", 3},
        {"agent:reserved_for_future_use", 4},
        {"converting", 5},
        {"episode_completion_pct", 7},
        {"last_action", 8},
        {"last_action_arg", 9},
        {"last_reward", 10},
        {"vibe", 11},
        {"agent:vibe", 12},
        {"agent:compass", 14},
        {"tag", 15},
        {"cooldown_remaining", 16},
        {"clipped", 17},
        {"remaining_uses", 18},
    };
    ObservationFeature::Initialize(feature_ids);
  }

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
  auto agent_at_location = grid.object_at(GridLocation(2, 3));
  EXPECT_EQ(agent_at_location, agent);
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

TEST_F(MettaGridCppTest, FractionalConsumptionChangeVibeAction) {
  Grid grid(3, 3);

  // Create agent with resources
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 30;
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(1, 1, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->init(&agent_reward);
  grid.add_object(agent);

  // Create change vibe action with fractional consumption (1.25)
  ChangeVibeActionConfig vibe_cfg({{TestItems::ORE, 2}}, {{TestItems::ORE, 1.25f}}, 4);
  GameConfig game_config;
  ChangeVibe change_vibe(vibe_cfg, &game_config);
  std::mt19937 rng(42);
  change_vibe.init(&grid, &rng);

  // Change vibe multiple times
  int changes = 0;
  ObservationType initial_vibe = agent->vibe;
  while (agent->inventory.amount(TestItems::ORE) >= 2) {
    bool success = change_vibe.handle_action(*agent, (initial_vibe + 1) % 4);
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
  AssemblerConfig config(1, "test_assembler");
  config.tag_ids = {1, 2};
  Assembler assembler(5, 5, config);

  unsigned int current_timestep = 0;
  assembler.set_current_timestep_ptr(&current_timestep);

  auto features = assembler.obs_features();

  // Should have at least Tag features
  EXPECT_GE(features.size(), 2);

  bool found_tag1 = false;
  bool found_tag2 = false;

  for (const auto& feature : features) {
    if (feature.feature_id == ObservationFeature::Tag) {
      if (feature.value == 1) {
        found_tag1 = true;
      } else if (feature.value == 2) {
        found_tag2 = true;
      }
    }
  }

  EXPECT_TRUE(found_tag1) << "Tag 1 not found";
  EXPECT_TRUE(found_tag2) << "Tag 2 not found";
}

TEST_F(MettaGridCppTest, AssemblerNoCooldownObservation) {
  AssemblerConfig config(1, "test_assembler");
  config.tag_ids = {1, 2};
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
  AssemblerConfig config(1, "test_assembler");
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
  AssemblerConfig config(1, "test_assembler");
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
  AssemblerConfig config(1, "test_assembler");
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

TEST_F(MettaGridCppTest, AssemblerGetCurrentProtocol) {
  // Create a grid to test with
  Grid grid(10, 10);

  AssemblerConfig config(1, "test_assembler");
  config.tag_ids = {1, 2};

  // Create test protocols
  auto protocol0 = std::make_shared<Protocol>();  // Default protocol (vibe 0)
  protocol0->input_resources[0] = 1;

  auto protocol1 = std::make_shared<Protocol>(std::vector<ObservationType>{1});  // Protocol for vibe 1
  protocol1->input_resources[1] = 2;

  config.protocols.push_back(protocol0);
  config.protocols.push_back(protocol1);
  Assembler* assembler = new Assembler(5, 5, config);

  // Set up the assembler with grid and timestep
  unsigned int current_timestep = 0;
  assembler->set_current_timestep_ptr(&current_timestep);
  assembler->set_grid(&grid);

  // Add assembler to grid
  grid.add_object(assembler);

  // Without agents around, should get protocol0
  const Protocol* current_protocol = assembler->get_current_protocol();
  EXPECT_EQ(current_protocol, protocol0.get());

  // With one agent and no vibe, should still get 0
  AgentConfig agent_cfg(1, "test_agent", 0, "test_group");
  auto resource_names = create_test_resource_names();
  Agent* agent = new Agent(4, 4, agent_cfg, &resource_names);  // NW of assembler
  grid.add_object(agent);

  current_protocol = assembler->get_current_protocol();
  EXPECT_EQ(current_protocol, protocol0.get()) << "With one agent, should still get protocol0";

  // Now with a vibe, should get protocol1
  agent->vibe = 1;

  current_protocol = assembler->get_current_protocol();
  EXPECT_EQ(current_protocol, protocol1.get()) << "With one agent and a vibe, should get protocol1";
}

TEST_F(MettaGridCppTest, AssemblerProtocolObservationsEnabled) {
  // Create a grid to test with
  Grid grid(10, 10);

  AssemblerConfig config(1, "test_assembler");

  // Create test protocols - one for pattern 0 (no agents), one for pattern 1 (some agents)
  auto protocol0 = std::make_shared<Protocol>();  // Default protocol (vibe 0)
  protocol0->input_resources[0] = 2;              // 2 units of item 0
  protocol0->output_resources[1] = 1;             // 1 unit of output item 1

  auto protocol1 = std::make_shared<Protocol>(std::vector<ObservationType>{1});  // Protocol for vibe 1
  protocol1->input_resources[2] = 3;                                             // 3 units of item 2
  protocol1->output_resources[3] = 2;                                            // 2 units of output item 3

  config.protocols.push_back(protocol0);
  config.protocols.push_back(protocol1);

  Assembler* assembler = new Assembler(5, 5, config);

  // Set up the assembler with grid and timestep
  unsigned int current_timestep = 0;
  assembler->set_current_timestep_ptr(&current_timestep);
  assembler->set_grid(&grid);

  // Add assembler to grid
  grid.add_object(assembler);

  // Provide an ObservationEncoder so protocol details can be emitted
  auto resource_names = create_test_resource_names();
  std::unordered_map<std::string, ObservationType> proto_feature_ids;
  // Assign arbitrary, unique feature ids for protocol input/output per resource
  for (size_t i = 0; i < resource_names.size(); ++i) {
    proto_feature_ids[std::string("protocol_input:") + resource_names[i]] = static_cast<ObservationType>(100 + i);
    proto_feature_ids[std::string("protocol_output:") + resource_names[i]] = static_cast<ObservationType>(120 + i);
    proto_feature_ids[std::string("inv:") + resource_names[i]] = static_cast<ObservationType>(140 + i);
  }
  ObservationEncoder encoder(true, resource_names, proto_feature_ids);
  assembler->set_obs_encoder(&encoder);

  // Test with pattern 0 (no agents around) - should get protocol0
  auto features = assembler->obs_features();

  // Should have protocol features - check that we have features but don't check specific IDs
  // since input_protocol_offset and output_protocol_offset are not in AssemblerConfig
  EXPECT_GT(features.size(), 0) << "Should have observation features";

  // Verify we're getting the right protocol
  const Protocol* current_protocol = assembler->get_current_protocol();
  EXPECT_EQ(current_protocol, protocol0.get());
}

TEST_F(MettaGridCppTest, AssemblerBalancedConsumptionAmpleResources) {
  // Test case (a): 3 agents with ample resources, consume 10 total
  // Each agent should lose 3-4 resources for balanced consumption

  // Create a protocol that requires 10 ore
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  input_resources[TestItems::ORE] = 10;

  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  output_resources[TestItems::LASER] = 1;

  auto protocol = std::make_shared<Protocol>(std::vector<ObservationType>{}, input_resources, output_resources, 0);

  // Create assembler with the protocol
  AssemblerConfig config(1, "test_assembler");
  config.protocols.push_back(protocol);
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
  assembler.consume_resources_for_protocol(*protocol, surrounding_agents);

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

  // Create a protocol that requires 20 ore
  std::unordered_map<InventoryItem, InventoryQuantity> input_resources;
  input_resources[TestItems::ORE] = 20;

  std::unordered_map<InventoryItem, InventoryQuantity> output_resources;
  output_resources[TestItems::LASER] = 1;

  auto protocol = std::make_shared<Protocol>(std::vector<ObservationType>{}, input_resources, output_resources, 0);

  // Create assembler with the protocol
  AssemblerConfig config(1, "test_assembler");
  config.protocols.push_back(protocol);
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
  assembler.consume_resources_for_protocol(*protocol, surrounding_agents);

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

  // Create an assembler with normal protocols
  AssemblerConfig config(1, "test_assembler");

  // Create normal protocols (pattern 0: no agents needed)
  auto normal_protocol = std::make_shared<Protocol>();
  normal_protocol->input_resources[TestItems::ORE] = 2;
  normal_protocol->output_resources[TestItems::LASER] = 1;
  normal_protocol->cooldown = 0;

  config.protocols.push_back(normal_protocol);

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

  // Test 2: Verify normal protocol works when not clipped
  bool success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use normal protocol when not clipped";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 8) << "Should consume 2 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 1) << "Should produce 1 laser";

  // Test 3: Create unclipping protocols and clip the assembler
  auto unclip_protocol = std::make_shared<Protocol>();
  unclip_protocol->input_resources[TestItems::HEART] = 1;
  unclip_protocol->output_resources[TestItems::ORE] = 3;
  unclip_protocol->cooldown = 0;

  std::vector<std::shared_ptr<Protocol>> unclip_protocols = {unclip_protocol};
  assembler.become_clipped(unclip_protocols, nullptr);

  EXPECT_TRUE(assembler.is_clipped) << "Assembler should be clipped after become_clipped()";

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

  // Test 5: Verify unclip protocol is used when clipped
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use unclip protocol when clipped";
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 4) << "Should consume 1 heart for unclipping";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 11) << "Should produce 3 ore from unclip protocol";

  // Test 6: Verify assembler is automatically unclipped after successful use
  EXPECT_FALSE(assembler.is_clipped) << "Assembler should be unclipped after successful use";
  EXPECT_TRUE(assembler.unclip_protocols.empty()) << "Unclip protocols should be empty";

  // Test 7: Verify normal protocol works again after unclipping
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should be able to use normal protocol after unclipping";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 9) << "Should consume 2 ore (normal protocol)";
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
  AssemblerConfig config(1, "test_assembler");
  config.max_uses = 3;  // Limit to 3 uses

  // Create simple protocol
  auto protocol = std::make_shared<Protocol>();
  protocol->input_resources[TestItems::ORE] = 1;
  protocol->output_resources[TestItems::LASER] = 1;
  protocol->cooldown = 0;

  config.protocols.push_back(protocol);

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

TEST_F(MettaGridCppTest, AssemblerWontProduceOutputIfAgentsCantReceive) {
  // Create a simple grid
  Grid grid(10, 10);
  unsigned int current_timestep = 0;

  // Create an assembler with a protocol that produces output
  AssemblerConfig config(1, "test_assembler");
  auto protocol = std::make_shared<Protocol>();
  protocol->input_resources[TestItems::ORE] = 2;
  protocol->output_resources[TestItems::LASER] = 1;  // Produces 1 LASER
  protocol->cooldown = 0;

  config.protocols.push_back(protocol);

  Assembler assembler(5, 5, config);
  assembler.set_grid(&grid);
  assembler.set_current_timestep_ptr(&current_timestep);

  auto resource_names = create_test_resource_names();

  // Create a single agent that we'll reuse for different tests
  AgentConfig agent_cfg = create_test_agent_config();
  agent_cfg.initial_inventory[TestItems::ORE] = 10;  // Has input resources

  Agent* agent = new Agent(4, 5, agent_cfg, &resource_names);
  float agent_reward = 0.0f;
  agent->reward = &agent_reward;
  grid.add_object(agent);

  // Test 1: Agent with full inventory for output item - should fail
  agent->update_inventory(TestItems::LASER, 50);  // Fill to limit (50)

  // Verify agent has full inventory
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 50);
  EXPECT_EQ(agent->inventory.free_space(TestItems::LASER), 0);

  // Try to use assembler - should fail because agent can't receive output
  bool success = assembler.onUse(*agent, 0);
  EXPECT_FALSE(success) << "Should fail when agents can't receive output";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 10) << "Input resources should not be consumed";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 50) << "Output should not be produced";

  // Test 2: Agent with space for output - should succeed
  // Remove all LASER to make space
  agent->update_inventory(TestItems::LASER, -50);
  // ORE should still be 10 since Test 1 failed and didn't consume it

  // Verify agent has space
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 0);
  EXPECT_GT(agent->inventory.free_space(TestItems::LASER), 0);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 10);

  // Try to use assembler - should succeed
  success = assembler.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should succeed when agents can receive output";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 8) << "Should consume 2 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 1) << "Should produce 1 laser";

  // Test 3: Protocol with no output - should always succeed (if inputs are available)
  AssemblerConfig config_no_output(1, "test_assembler_no_output");
  auto protocol_no_output = std::make_shared<Protocol>();
  protocol_no_output->input_resources[TestItems::ORE] = 1;
  // No output resources
  protocol_no_output->cooldown = 0;

  config_no_output.protocols.push_back(protocol_no_output);

  Assembler assembler_no_output(5, 5, config_no_output);
  assembler_no_output.set_grid(&grid);
  assembler_no_output.set_current_timestep_ptr(&current_timestep);

  // Fill agent's inventory again
  agent->update_inventory(TestItems::LASER, 50);  // Fill to limit
  // Calculate delta needed to get ORE back to 10 (it was consumed to 8 in Test 2)
  agent->update_inventory(TestItems::ORE, 2);  // Reset ORE to 10

  // Agent with full inventory - should still succeed because protocol has no output
  success = assembler_no_output.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should succeed when protocol has no output, even if inventory is full";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 9) << "Should consume 1 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 50) << "Output should remain unchanged";

  // Test 4: Multiple agents, all with full inventory - should fail
  Assembler assembler_multi(5, 5, config);
  assembler_multi.set_grid(&grid);
  assembler_multi.set_current_timestep_ptr(&current_timestep);

  // Reset agent's state
  agent->update_inventory(TestItems::ORE, 10);
  agent->update_inventory(TestItems::LASER, 50);  // Full

  // Add a second agent at a different surrounding position
  AgentConfig agent_cfg2 = create_test_agent_config();
  agent_cfg2.initial_inventory[TestItems::ORE] = 10;
  agent_cfg2.initial_inventory[TestItems::LASER] = 50;  // Full

  Agent* agent2 = new Agent(4, 6, agent_cfg2, &resource_names);
  float reward2 = 0.0f;
  agent2->reward = &reward2;
  grid.add_object(agent2);

  // Both agents have full inventory
  EXPECT_EQ(agent->inventory.free_space(TestItems::LASER), 0);
  EXPECT_EQ(agent2->inventory.free_space(TestItems::LASER), 0);

  success = assembler_multi.onUse(*agent, 0);
  EXPECT_FALSE(success) << "Should fail when all surrounding agents can't receive output";
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
                               1,                           // agent_radius
                               false);                      // scales
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

// Tests for HasInventory::shared_update function
TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_EvenDistribution) {
  // Test that positive delta is evenly distributed among agents
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Add 30 ore, should be distributed as 10 each
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_UnevenDistribution) {
  // Test that when delta doesn't divide evenly, earlier agents get more
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Add 31 ore, should be distributed as 11, 10, 10 (earlier agents get more)
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 31);

  EXPECT_EQ(consumed, 31);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 11);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_WithLimits) {
  // Test that agents that hit their inventory limit drop out of distribution
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 10}};  // Low limit of 10

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent1 with 5 ore
  agent1.update_inventory(TestItems::ORE, 5);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Try to add 30 ore
  // agent1 can only take 5 more (to reach limit of 10)
  // agent2 and agent3 can each take 10 (to reach their limits)
  // Total consumed will be 5 + 10 + 10 = 25, not the full 30
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 25);                                 // Only 25 can be consumed due to limits
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);  // Hit limit
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_EvenDistribution) {
  // Test that negative delta is evenly distributed among agents
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with 20 ore each
  agent1.update_inventory(TestItems::ORE, 20);
  agent2.update_inventory(TestItems::ORE, 20);
  agent3.update_inventory(TestItems::ORE, 20);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Remove 30 ore, should remove 10 from each
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_InsufficientResources) {
  // Test behavior when some agents don't have enough to contribute their share
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with different amounts
  agent1.update_inventory(TestItems::ORE, 5);   // Only has 5
  agent2.update_inventory(TestItems::ORE, 20);  // Has plenty
  agent3.update_inventory(TestItems::ORE, 20);  // Has plenty

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Try to remove 30 ore
  // agent1 can only contribute 5, remaining 25 split between agent2 and agent3 as 13, 12
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 0);  // Depleted
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 7);  // 20 - 13
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 8);  // 20 - 12
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_UnevenDistribution) {
  // Test that when negative delta doesn't divide evenly, earlier agents lose more
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with 20 ore each
  agent1.update_inventory(TestItems::ORE, 20);
  agent2.update_inventory(TestItems::ORE, 20);
  agent3.update_inventory(TestItems::ORE, 20);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Remove 31 ore, should remove 11, 10, 10 (earlier agents lose more)
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, -31);

  EXPECT_EQ(consumed, -31);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 9);   // 20 - 11
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);  // 20 - 10
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);  // 20 - 10
}

TEST_F(MettaGridCppTest, SharedUpdate_EmptyInventoriesList) {
  // Test with empty inventory havers list
  std::vector<HasInventory*> inventory_havers;

  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 10);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since no inventory havers
}

TEST_F(MettaGridCppTest, SharedUpdate_SingleInventory) {
  // Test with single agent
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 100}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  std::vector<HasInventory*> inventory_havers = {&agent1};

  // All delta should go to the single agent
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 25);

  EXPECT_EQ(consumed, 25);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 25);
}

TEST_F(MettaGridCppTest, SharedUpdate_AllInventoriesAtLimit) {
  // Test when all agent inventories are at their limit
  InventoryConfig inv_cfg;
  inv_cfg.limits = {{{TestItems::ORE}, 10}};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);

  // Fill both to limit
  agent1.update_inventory(TestItems::ORE, 10);
  agent2.update_inventory(TestItems::ORE, 10);

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2};

  // Try to add more
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 20);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since all at limit
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_MixedLimits) {
  // Test with agents having different inventory limits
  InventoryConfig inv_cfg1;
  inv_cfg1.limits = {{{TestItems::ORE}, 10}};

  InventoryConfig inv_cfg2;
  inv_cfg2.limits = {{{TestItems::ORE}, 20}};

  InventoryConfig inv_cfg3;
  inv_cfg3.limits = {{{TestItems::ORE}, 30}};

  AgentConfig agent_cfg1(1, "test_agent1", 1, "test_group");
  agent_cfg1.inventory_config = inv_cfg1;

  AgentConfig agent_cfg2(2, "test_agent2", 1, "test_group");
  agent_cfg2.inventory_config = inv_cfg2;

  AgentConfig agent_cfg3(3, "test_agent3", 1, "test_group");
  agent_cfg3.inventory_config = inv_cfg3;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg1, &resource_names);  // Limit 10
  Agent agent2(1, 0, agent_cfg2, &resource_names);  // Limit 20
  Agent agent3(2, 0, agent_cfg3, &resource_names);  // Limit 30

  std::vector<HasInventory*> inventory_havers = {&agent1, &agent2, &agent3};

  // Try to add 45 ore
  // agent1 takes 10 (hits limit), agent2 takes 18, agent3 takes 17
  InventoryDelta consumed = HasInventory::shared_update(inventory_havers, TestItems::ORE, 45);

  EXPECT_EQ(consumed, 45);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 18);  // Gets more due to being earlier
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 17);
}
