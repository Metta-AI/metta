#include <gtest/gtest.h>

#include <array>
#include <random>
#include <utility>

#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "actions/noop.hpp"
#include "config/mettagrid_config.hpp"
#include "config/observation_features.hpp"
#include "core/grid.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/agent_config.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/constants.hpp"
#include "objects/inventory.hpp"
#include "objects/inventory_config.hpp"
#include "objects/protocol.hpp"
#include "objects/wall.hpp"
#include "systems/stats_tracker.hpp"

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
        {"episode_completion_pct", 7},
        {"last_action", 8},
        {"goal", 9},
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
    resource_names = create_test_resource_names();
    stats_tracker = std::make_unique<StatsTracker>(&resource_names);
  }

  void TearDown() override {}

  // Helper function to create test resource_limits map
  InventoryConfig create_test_inventory_config() {
    InventoryConfig inventory_config;
    inventory_config.limit_defs = {
        LimitDef({TestItems::ORE}, 50),
        LimitDef({TestItems::LASER}, 50),
        LimitDef({TestItems::ARMOR}, 50),
        LimitDef({TestItems::HEART}, 50),
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
                       0,                               // initial_vibe
                       create_test_inventory_config(),  // inventory_config
                       create_test_stats_rewards(),     // stat_rewards
                       create_test_stats_reward_max(),  // stat_reward_max
                       {});                             // initial_inventory
  }

  std::vector<std::string> resource_names;
  std::unique_ptr<StatsTracker> stats_tracker;
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

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0, create_test_inventory_config(), rewards, stats_reward_max);
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
  int delta = agent->inventory.update(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.625f);  // 5 * 0.125

  // Test removing items
  delta = agent->inventory.update(TestItems::ORE, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 3);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.375f);  // 3 * 0.125

  // Test hitting zero
  delta = agent->inventory.update(TestItems::ORE, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  // check that the item is not in the inventory
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 0.0f);

  // Test hitting resource_limits limit
  agent->inventory.update(TestItems::ORE, 30);
  delta = agent->inventory.update(TestItems::ORE, 50);  // resource_limits is 50
  EXPECT_EQ(delta, 20);                                 // Should only add up to resource_limits
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 50);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 6.25f);  // 50 * 0.125
}

TEST_F(MettaGridCppTest, AgentInventoryStatsUpdate) {
  AgentConfig agent_cfg = create_test_agent_config();
  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));

  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test that stats are updated when inventory changes via inventory.update() directly
  // This verifies the on_inventory_change callback mechanism

  // Initial state: no stats should be set
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 0.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 0.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 0.0f);

  InventoryDelta delta1 = agent->inventory.update(TestItems::ORE, 10);
  EXPECT_EQ(delta1, 10);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 10);

  // Verify stats were updated via callback
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 10.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 10.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 0.0f);

  // Add more items
  InventoryDelta delta2 = agent->inventory.update(TestItems::ORE, 5);
  EXPECT_EQ(delta2, 5);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 15);

  // Verify stats accumulated correctly
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 15.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 15.0f);  // 10 + 5
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 0.0f);

  // Remove items
  InventoryDelta delta3 = agent->inventory.update(TestItems::ORE, -7);
  EXPECT_EQ(delta3, -7);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 8);

  // Verify stats updated correctly
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 8.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 15.0f);  // Unchanged
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 7.0f);     // 0 + 7

  // Remove more items
  InventoryDelta delta4 = agent->inventory.update(TestItems::ORE, -3);
  EXPECT_EQ(delta4, -3);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);

  // Verify stats updated correctly
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 5.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 15.0f);  // Unchanged
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 10.0f);    // 7 + 3

  // Test with a different resource (LASER)
  InventoryDelta delta5 = agent->inventory.update(TestItems::LASER, 20);
  EXPECT_EQ(delta5, 20);
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 20);

  // Verify LASER stats were updated
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::LASER) + ".amount"), 20.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::LASER) + ".gained"), 20.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::LASER) + ".lost"), 0.0f);

  // Test that zero delta doesn't update stats (but callback should still be called)
  InventoryDelta delta6 = agent->inventory.update(TestItems::ORE, 0);
  EXPECT_EQ(delta6, 0);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);

  // Stats should remain unchanged (delta was 0, so no stats update in callback)
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 5.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"), 15.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 10.0f);

  // Test hitting limit - stats should reflect actual change, not attempted change
  agent->inventory.update(TestItems::ORE, 50);                           // Fill to limit
  InventoryDelta delta7 = agent->inventory.update(TestItems::ORE, 100);  // Try to add 100, but limit is 50
  EXPECT_EQ(delta7, 0);                                                  // No change because already at limit
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 50);

  // Stats should reflect only the actual change (0), so no update
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".amount"), 50.0f);
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".gained"),
                  60.0f);  // 15 + 45 (from filling to limit)
  EXPECT_FLOAT_EQ(agent->stats.get(std::string(TestItemStrings::ORE) + ".lost"), 10.0f);
}
// Test for reward capping behavior with a lower cap to actually hit it
TEST_F(MettaGridCppTest, AgentInventoryUpdate_RewardCappingBehavior) {
  // Create a custom config with a lower ore reward cap that we can actually hit
  auto inventory_config = create_test_inventory_config();
  auto rewards = create_test_stats_rewards();

  // Set a lower cap for ORE so we can actually test capping
  std::unordered_map<std::string, RewardType> stats_reward_max;
  stats_reward_max[std::string(TestItemStrings::ORE) + ".amount"] = 2.0f;  // Cap at 2.0 instead of 10.0

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0, inventory_config, rewards, stats_reward_max);

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Test 1: Add items up to the cap
  // 16 ORE * 0.125 = 2.0 (exactly at cap)
  int delta = agent->inventory.update(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 16);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Test 2: Add more items beyond the cap
  // 32 ORE * 0.125 = 4.0, but capped at 2.0
  delta = agent->inventory.update(TestItems::ORE, 16);
  EXPECT_EQ(delta, 16);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 32);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Still capped at 2.0

  // Test 3: Remove some items while still over cap
  // 24 ORE * 0.125 = 3.0, but still capped at 2.0
  delta = agent->inventory.update(TestItems::ORE, -8);
  EXPECT_EQ(delta, -8);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 24);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);  // Should remain at cap

  // Test 4: Remove enough items to go below cap
  // 12 ORE * 0.125 = 1.5 (now below cap)
  delta = agent->inventory.update(TestItems::ORE, -12);
  EXPECT_EQ(delta, -12);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 12);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 1.5f);  // Now tracking actual value

  // Test 5: Add items again, but not enough to hit cap
  // 14 ORE * 0.125 = 1.75 (still below cap)
  delta = agent->inventory.update(TestItems::ORE, 2);
  EXPECT_EQ(delta, 2);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 14);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 1.75f);

  // Test 6: Add items to go over cap again
  // 20 ORE * 0.125 = 2.5, but capped at 2.0
  delta = agent->inventory.update(TestItems::ORE, 6);
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

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0, inventory_config, rewards, stats_reward_max);

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add ORE beyond its cap
  agent->inventory.update(TestItems::ORE, 50);  // 50 * 0.125 = 6.25, capped at 2.0
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 50);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 2.0f);

  // Add HEART up to its cap
  agent->inventory.update(TestItems::HEART, 30);  // 30 * 1.0 = 30.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 30);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // 2.0 + 30.0

  // Add more HEART beyond its cap
  agent->inventory.update(TestItems::HEART, 10);  // 40 * 1.0 = 40.0, capped at 30.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 40);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // Still 2.0 + 30.0

  // Remove some ORE (still over cap)
  agent->inventory.update(TestItems::ORE, -10);  // 40 * 0.125 = 5.0, still capped at 2.0
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 40);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 32.0f);  // No change

  // Remove ORE to go below cap
  agent->inventory.update(TestItems::ORE, -35);  // 5 * 0.125 = 0.625
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 5);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 30.625f);  // 0.625 + 30.0

  // Remove HEART to go below its cap
  agent->inventory.update(TestItems::HEART, -15);  // 25 * 1.0 = 25.0
  EXPECT_EQ(agent->inventory.amount(TestItems::HEART), 25);
  agent->compute_stat_rewards();
  EXPECT_FLOAT_EQ(agent_reward, 25.625f);  // 0.625 + 25.0
}

// Test shared inventory limits between multiple resources
TEST_F(MettaGridCppTest, SharedInventoryLimits) {
  // Create an inventory config where ORE and LASER share a combined limit
  InventoryConfig inventory_config;
  inventory_config.limit_defs = {
      LimitDef({TestItems::ORE, TestItems::LASER}, 30),  // ORE and LASER share a limit of 30 total
      LimitDef({TestItems::ARMOR}, 50),                  // ARMOR has its own separate limit
      {{TestItems::HEART}, 50},                          // HEART has its own separate limit
  };

  auto rewards = create_test_stats_rewards();
  auto stats_reward_max = create_test_stats_reward_max();

  AgentConfig agent_cfg(0, "agent", 1, "test_group", 100, 0, inventory_config, rewards, stats_reward_max);

  auto resource_names = create_test_resource_names();
  std::unique_ptr<Agent> agent(new Agent(0, 0, agent_cfg, &resource_names));
  float agent_reward = 0.0f;
  agent->init(&agent_reward);

  // Add ORE up to 20
  int delta = agent->inventory.update(TestItems::ORE, 20);
  EXPECT_EQ(delta, 20);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 20);

  // Try to add 20 LASER - should only add 10 due to shared limit
  delta = agent->inventory.update(TestItems::LASER, 20);
  EXPECT_EQ(delta, 10);  // Only 10 can be added (20 ORE + 10 LASER = 30 total)
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 10);

  // Try to add more ORE - should fail as we're at the shared limit
  delta = agent->inventory.update(TestItems::ORE, 5);
  EXPECT_EQ(delta, 0);  // Can't add any more
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 20);

  // Remove some LASER
  delta = agent->inventory.update(TestItems::LASER, -5);
  EXPECT_EQ(delta, -5);
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 5);

  // Now we can add more ORE since we freed up shared space
  delta = agent->inventory.update(TestItems::ORE, 5);
  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 25);

  // ARMOR should work independently with its own limit
  delta = agent->inventory.update(TestItems::ARMOR, 40);
  EXPECT_EQ(delta, 40);
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 40);

  // Can still add more ARMOR up to its limit
  delta = agent->inventory.update(TestItems::ARMOR, 20);
  EXPECT_EQ(delta, 10);  // Should cap at 50
  EXPECT_EQ(agent->inventory.amount(TestItems::ARMOR), 50);

  // Remove all ORE
  delta = agent->inventory.update(TestItems::ORE, -25);
  EXPECT_EQ(delta, -25);
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 0);

  // Now we can add up to 25 more LASER (5 existing + 25 = 30)
  delta = agent->inventory.update(TestItems::LASER, 30);
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

// ==================== Assembler Tests ====================

TEST_F(MettaGridCppTest, AssemblerBasicObservationFeatures) {
  AssemblerConfig config(1, "test_assembler");
  config.tag_ids = {1, 2};
  Assembler assembler(5, 5, config, stats_tracker.get());

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
  Assembler assembler(5, 5, config, stats_tracker.get());

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
  Assembler assembler(5, 5, config, stats_tracker.get());

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
  Assembler assembler(5, 5, config, stats_tracker.get());

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
  Assembler assembler(5, 5, config, stats_tracker.get());

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

  auto protocol1 = std::make_shared<Protocol>(0, std::vector<ObservationType>{1});  // Protocol for vibe 1
  protocol1->input_resources[1] = 2;

  config.protocols.push_back(protocol0);
  config.protocols.push_back(protocol1);
  Assembler* assembler = new Assembler(5, 5, config, stats_tracker.get());

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

  auto protocol1 = std::make_shared<Protocol>(0, std::vector<ObservationType>{1});  // Protocol for vibe 1
  protocol1->input_resources[2] = 3;                                                // 3 units of item 2
  protocol1->output_resources[3] = 2;                                               // 2 units of output item 3

  config.protocols.push_back(protocol0);
  config.protocols.push_back(protocol1);

  Assembler* assembler = new Assembler(5, 5, config, stats_tracker.get());

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
  // Use multi-token encoding with :p1, :p2 suffixes (default token_value_base=256 needs 2 tokens for uint16_t)
  for (size_t i = 0; i < resource_names.size(); ++i) {
    proto_feature_ids[std::string("protocol_input:") + resource_names[i]] = static_cast<ObservationType>(100 + i);
    proto_feature_ids[std::string("protocol_output:") + resource_names[i]] = static_cast<ObservationType>(120 + i);
    proto_feature_ids[std::string("inv:") + resource_names[i]] = static_cast<ObservationType>(140 + i);
    proto_feature_ids[std::string("inv:") + resource_names[i] + ":p1"] = static_cast<ObservationType>(160 + i);
    proto_feature_ids[std::string("inv:") + resource_names[i] + ":p2"] = static_cast<ObservationType>(180 + i);
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

  auto protocol = std::make_shared<Protocol>(0, std::vector<ObservationType>{}, input_resources, output_resources, 0);

  // Create assembler with the protocol
  AssemblerConfig config(1, "test_assembler");
  config.protocols.push_back(protocol);
  Assembler assembler(5, 5, config, stats_tracker.get());

  // Create agents with ample resources
  AgentConfig agent_config(0, "agent", 0, "agent");
  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_config, &resource_names);
  Agent agent2(0, 0, agent_config, &resource_names);
  Agent agent3(0, 0, agent_config, &resource_names);

  agent1.inventory.update(TestItems::ORE, 20);
  agent2.inventory.update(TestItems::ORE, 20);
  agent3.inventory.update(TestItems::ORE, 20);

  std::vector<Agent*> surrounding_agents = {&agent1, &agent2, &agent3};
  // Extract Inventory* pointers from agents for consume_resources_for_protocol
  std::vector<Inventory*> surrounding_inventories;
  for (Agent* agent : surrounding_agents) {
    surrounding_inventories.push_back(&agent->inventory);
  }

  // Consume resources
  assembler.consume_resources_for_protocol(*protocol, surrounding_inventories);

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

  auto protocol = std::make_shared<Protocol>(0, std::vector<ObservationType>{}, input_resources, output_resources, 0);

  // Create assembler with the protocol
  AssemblerConfig config(1, "test_assembler");
  config.protocols.push_back(protocol);
  Assembler assembler(5, 5, config, stats_tracker.get());

  // Create agents with varied resources
  AgentConfig agent_config(0, "agent", 0, "agent");
  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_config, &resource_names);
  Agent agent2(0, 0, agent_config, &resource_names);
  Agent agent3(0, 0, agent_config, &resource_names);
  Agent agent4(0, 0, agent_config, &resource_names);

  agent1.inventory.update(TestItems::ORE, 0);   // No resources
  agent2.inventory.update(TestItems::ORE, 1);   // Limited resources
  agent3.inventory.update(TestItems::ORE, 50);  // Ample resources
  agent4.inventory.update(TestItems::ORE, 50);  // Ample resources

  std::vector<Agent*> surrounding_agents = {&agent1, &agent2, &agent3, &agent4};
  // Extract Inventory* pointers from agents for consume_resources_for_protocol
  std::vector<Inventory*> surrounding_inventories;
  for (Agent* agent : surrounding_agents) {
    surrounding_inventories.push_back(&agent->inventory);
  }

  // Consume resources
  assembler.consume_resources_for_protocol(*protocol, surrounding_inventories);

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

  Assembler assembler(5, 5, config, stats_tracker.get());
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

  Assembler assembler(5, 5, config, stats_tracker.get());
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

TEST_F(MettaGridCppTest, AssemblerMinAgentsProtocolSelection) {
  // Create a simple grid
  Grid grid(10, 10);
  unsigned int current_timestep = 0;

  // Create an assembler with multiple protocols with different min_agents values
  AssemblerConfig config(1, "test_assembler");

  // Protocol 1: min_agents = 0 (can be used with any number of agents)
  auto protocol_0 = std::make_shared<Protocol>(0, std::vector<ObservationType>{});
  protocol_0->input_resources[TestItems::ORE] = 1;
  protocol_0->output_resources[TestItems::LASER] = 1;
  protocol_0->cooldown = 0;

  // Protocol 2: min_agents = 2 (requires at least 2 agents)
  auto protocol_2 = std::make_shared<Protocol>(2, std::vector<ObservationType>{});
  protocol_2->input_resources[TestItems::ORE] = 2;
  protocol_2->output_resources[TestItems::ARMOR] = 1;
  protocol_2->cooldown = 0;

  // Protocol 3: min_agents = 4 (requires at least 4 agents)
  auto protocol_4 = std::make_shared<Protocol>(4, std::vector<ObservationType>{});
  protocol_4->input_resources[TestItems::ORE] = 3;
  protocol_4->output_resources[TestItems::HEART] = 1;
  protocol_4->cooldown = 0;

  // Add protocols in order (they should be sorted by min_agents descending)
  config.protocols.push_back(protocol_0);
  config.protocols.push_back(protocol_2);
  config.protocols.push_back(protocol_4);

  Assembler assembler(5, 5, config, stats_tracker.get());
  assembler.set_grid(&grid);
  assembler.set_current_timestep_ptr(&current_timestep);

  auto resource_names = create_test_resource_names();

  // Positions around (5, 5): (4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)
  std::vector<std::pair<GridCoord, GridCoord>> positions = {
      {4, 4}, {4, 5}, {4, 6}, {5, 4}, {5, 6}, {6, 4}, {6, 5}, {6, 6}};

  // Helper function to count and place agents around the assembler
  // Returns the total number of agents after placement
  auto place_agents = [&](int target_count) -> int {
    int current_count = 0;
    // Count existing agents
    for (const auto& pos : positions) {
      GridObject* obj = grid.object_at(GridLocation(pos.first, pos.second));
      if (obj) {
        Agent* agent = dynamic_cast<Agent*>(obj);
        if (agent) {
          current_count++;
        }
      }
    }
    // Add agents until we reach target_count
    for (int i = 0; current_count < target_count && i < static_cast<int>(positions.size()); ++i) {
      // Check if position is empty
      if (grid.is_empty(positions[i].first, positions[i].second)) {
        AgentConfig agent_cfg = create_test_agent_config();
        agent_cfg.initial_inventory[TestItems::ORE] = 10;
        Agent* agent = new Agent(positions[i].first, positions[i].second, agent_cfg, &resource_names);
        float agent_reward = 0.0f;
        agent->reward = &agent_reward;
        grid.add_object(agent);
        current_count++;
      }
    }
    return current_count;
  };

  // Test 1: With 0 agents, should return protocol_0 (min_agents = 0)
  const Protocol* current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 0 agents";
  EXPECT_EQ(current_protocol->min_agents, 0) << "Should return protocol with min_agents = 0";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::LASER), 1)
      << "Should return protocol_0 (produces LASER)";

  // Test 2: With 1 agent, should return protocol_0 (min_agents = 0)
  place_agents(1);
  current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 1 agent";
  EXPECT_EQ(current_protocol->min_agents, 0) << "Should return protocol with min_agents = 0";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::LASER), 1)
      << "Should return protocol_0 (produces LASER)";

  // Test 3: With 2 agents, should return protocol_2 (min_agents = 2, highest that fits)
  place_agents(2);
  current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 2 agents";
  EXPECT_EQ(current_protocol->min_agents, 2) << "Should return protocol with min_agents = 2";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::ARMOR), 1)
      << "Should return protocol_2 (produces ARMOR)";

  // Test 4: With 3 agents, should return protocol_2 (min_agents = 2, highest that fits)
  place_agents(3);
  current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 3 agents";
  EXPECT_EQ(current_protocol->min_agents, 2) << "Should return protocol with min_agents = 2";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::ARMOR), 1)
      << "Should return protocol_2 (produces ARMOR)";

  // Test 5: With 4 agents, should return protocol_4 (min_agents = 4, highest that fits)
  place_agents(4);
  current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 4 agents";
  EXPECT_EQ(current_protocol->min_agents, 4) << "Should return protocol with min_agents = 4";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::HEART), 1)
      << "Should return protocol_4 (produces HEART)";

  // Test 6: With 5 agents, should return protocol_4 (min_agents = 4, highest that fits)
  place_agents(5);
  current_protocol = assembler.get_current_protocol();
  EXPECT_NE(current_protocol, nullptr) << "Should return a protocol with 5 agents";
  EXPECT_EQ(current_protocol->min_agents, 4) << "Should return protocol with min_agents = 4";
  EXPECT_EQ(current_protocol->output_resources.count(TestItems::HEART), 1)
      << "Should return protocol_4 (produces HEART)";
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

  Assembler assembler(5, 5, config, stats_tracker.get());
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
  agent->inventory.update(TestItems::LASER, 50);  // Fill to limit (50)

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
  agent->inventory.update(TestItems::LASER, -50);
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

  Assembler assembler_no_output(5, 5, config_no_output, stats_tracker.get());
  assembler_no_output.set_grid(&grid);
  assembler_no_output.set_current_timestep_ptr(&current_timestep);

  // Fill agent's inventory again
  agent->inventory.update(TestItems::LASER, 50);  // Fill to limit
  // Calculate delta needed to get ORE back to 10 (it was consumed to 8 in Test 2)
  agent->inventory.update(TestItems::ORE, 2);  // Reset ORE to 10

  // Agent with full inventory - should still succeed because protocol has no output
  success = assembler_no_output.onUse(*agent, 0);
  EXPECT_TRUE(success) << "Should succeed when protocol has no output, even if inventory is full";
  EXPECT_EQ(agent->inventory.amount(TestItems::ORE), 9) << "Should consume 1 ore";
  EXPECT_EQ(agent->inventory.amount(TestItems::LASER), 50) << "Output should remain unchanged";

  // Test 4: Multiple agents, all with full inventory - should fail
  Assembler assembler_multi(5, 5, config, stats_tracker.get());
  assembler_multi.set_grid(&grid);
  assembler_multi.set_current_timestep_ptr(&current_timestep);

  // Reset agent's state
  agent->inventory.update(TestItems::ORE, 10);
  agent->inventory.update(TestItems::LASER, 50);  // Full

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

// Tests for HasInventory::shared_update function
TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_EvenDistribution) {
  // Test that positive delta is evenly distributed among agents
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Add 30 ore, should be distributed as 10 each
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_UnevenDistribution) {
  // Test that when delta doesn't divide evenly, earlier agents get more
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Add 31 ore, should be distributed as 11, 10, 10 (earlier agents get more)
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 31);

  EXPECT_EQ(consumed, 31);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 11);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_PositiveDelta_WithLimits) {
  // Test that agents that hit their inventory limit drop out of distribution
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 10)};  // Low limit of 10

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent1 with 5 ore
  agent1.inventory.update(TestItems::ORE, 5);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Try to add 30 ore
  // agent1 can only take 5 more (to reach limit of 10)
  // agent2 and agent3 can each take 10 (to reach their limits)
  // Total consumed will be 5 + 10 + 10 = 25, not the full 30
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 30);

  EXPECT_EQ(consumed, 25);                                 // Only 25 can be consumed due to limits
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);  // Hit limit
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_EvenDistribution) {
  // Test that negative delta is evenly distributed among agents
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with 20 ore each
  agent1.inventory.update(TestItems::ORE, 20);
  agent2.inventory.update(TestItems::ORE, 20);
  agent3.inventory.update(TestItems::ORE, 20);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Remove 30 ore, should remove 10 from each
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_InsufficientResources) {
  // Test behavior when some agents don't have enough to contribute their share
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with different amounts
  agent1.inventory.update(TestItems::ORE, 5);   // Only has 5
  agent2.inventory.update(TestItems::ORE, 20);  // Has plenty
  agent3.inventory.update(TestItems::ORE, 20);  // Has plenty

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Try to remove 30 ore
  // agent1 can only contribute 5, remaining 25 split between agent2 and agent3 as 13, 12
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, -30);

  EXPECT_EQ(consumed, -30);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 0);  // Depleted
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 7);  // 20 - 13
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 8);  // 20 - 12
}

TEST_F(MettaGridCppTest, SharedUpdate_NegativeDelta_UnevenDistribution) {
  // Test that when negative delta doesn't divide evenly, earlier agents lose more
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);
  Agent agent3(2, 0, agent_cfg, &resource_names);

  // Pre-fill agent inventories with 20 ore each
  agent1.inventory.update(TestItems::ORE, 20);
  agent2.inventory.update(TestItems::ORE, 20);
  agent3.inventory.update(TestItems::ORE, 20);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Remove 31 ore, should remove 11, 10, 10 (earlier agents lose more)
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, -31);

  EXPECT_EQ(consumed, -31);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 9);   // 20 - 11
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);  // 20 - 10
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 10);  // 20 - 10
}

TEST_F(MettaGridCppTest, SharedUpdate_EmptyInventoriesList) {
  // Test with empty inventory havers list
  std::vector<Inventory*> inventories;

  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 10);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since no inventory havers
}

TEST_F(MettaGridCppTest, SharedUpdate_SingleInventory) {
  // Test with single agent
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 100)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  std::vector<Inventory*> inventories = {&agent1.inventory};

  // All delta should go to the single agent
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 25);

  EXPECT_EQ(consumed, 25);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 25);
}

TEST_F(MettaGridCppTest, SharedUpdate_AllInventoriesAtLimit) {
  // Test when all agent inventories are at their limit
  InventoryConfig inv_cfg;
  inv_cfg.limit_defs = {LimitDef({TestItems::ORE}, 10)};

  AgentConfig agent_cfg(1, "test_agent", 1, "test_group");
  agent_cfg.inventory_config = inv_cfg;

  auto resource_names = create_test_resource_names();
  Agent agent1(0, 0, agent_cfg, &resource_names);
  Agent agent2(1, 0, agent_cfg, &resource_names);

  // Fill both to limit
  agent1.inventory.update(TestItems::ORE, 10);
  agent2.inventory.update(TestItems::ORE, 10);

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory};

  // Try to add more
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 20);

  EXPECT_EQ(consumed, 0);  // Nothing consumed since all at limit
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 10);
}

TEST_F(MettaGridCppTest, SharedUpdate_MixedLimits) {
  // Test with agents having different inventory limits
  InventoryConfig inv_cfg1;
  inv_cfg1.limit_defs = {LimitDef({TestItems::ORE}, 10)};

  InventoryConfig inv_cfg2;
  inv_cfg2.limit_defs = {LimitDef({TestItems::ORE}, 20)};

  InventoryConfig inv_cfg3;
  inv_cfg3.limit_defs = {LimitDef({TestItems::ORE}, 30)};

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

  std::vector<Inventory*> inventories = {&agent1.inventory, &agent2.inventory, &agent3.inventory};

  // Try to add 45 ore
  // agent1 takes 10 (hits limit), agent2 takes 18, agent3 takes 17
  InventoryDelta consumed = HasInventory::shared_update(inventories, TestItems::ORE, 45);

  EXPECT_EQ(consumed, 45);
  EXPECT_EQ(agent1.inventory.amount(TestItems::ORE), 10);  // Hit limit
  EXPECT_EQ(agent2.inventory.amount(TestItems::ORE), 18);  // Gets more due to being earlier
  EXPECT_EQ(agent3.inventory.amount(TestItems::ORE), 17);
}
