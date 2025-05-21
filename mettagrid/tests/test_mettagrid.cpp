#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "actions/attack.hpp"

namespace py = pybind11;

class MettaGridTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Python interpreter
    Py_Initialize();
  }

  void TearDown() override {
    // Finalize Python interpreter
    Py_Finalize();
  }

  // Helper function to create test configurations
  py::dict create_test_configs() {
    py::dict agent_cfg;
    agent_cfg["freeze_duration"] = 100;
    py::dict agent_rewards;
    agent_rewards["heart"] = 1.0;
    agent_rewards["ore.red"] = 0.125;  // Pick a power of 2 so floating point precision issues don't matter
    agent_cfg["rewards"] = agent_rewards;

    py::dict group_cfg;
    group_cfg["max_inventory"] = 123;
    py::dict group_rewards;
    group_rewards["ore.red"] = 0.0;    // Should override agent ore.red reward
    group_rewards["ore.green"] = 0.5;  // New reward
    group_cfg["rewards"] = group_rewards;

    py::dict configs;
    configs["agent_cfg"] = agent_cfg;
    configs["group_cfg"] = group_cfg;
    return configs;
  }
};

TEST_F(MettaGridTest, AgentCreation) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Test agent creation
  Agent* agent = MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);

  // Verify merged configuration
  EXPECT_EQ(agent->freeze_duration, 100);  // Group config overrides agent

  // Verify merged rewards
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::heart], 1.0);      // Agent reward preserved
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_red], 0.0);    // Group reward overrides
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_green], 0.5);  // Group reward added

  delete agent;
}

TEST_F(MettaGridTest, UpdateInventory) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  Agent* agent = MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Test adding items
  int delta = agent->update_inventory(InventoryItem::heart, 5);

  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 5);

  // Test removing items
  delta = agent->update_inventory(InventoryItem::heart, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 3);

  // Test hitting zero
  delta = agent->update_inventory(InventoryItem::heart, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 0);

  // Test hitting max_items limit
  agent->update_inventory(InventoryItem::heart, 50);
  delta = agent->update_inventory(InventoryItem::heart, 200);  // max_items is 123
  EXPECT_EQ(delta, 73);  // Should only add up to max_items
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 123);

  // Test multiple items
  delta = agent->update_inventory(InventoryItem::ore_red, 10);
  EXPECT_EQ(delta, 10);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 10);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 123);  // Other items unchanged

  delete agent;
}

TEST_F(MettaGridTest, AttackAction) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Create two agents - one attacker and one target
  Agent* attacker = MettaGrid::create_agent(2, 0, "red", 1, group_cfg, agent_cfg);
  Agent* target = MettaGrid::create_agent(0, 0, "blue", 2, group_cfg, agent_cfg);
  ASSERT_NE(attacker, nullptr);
  ASSERT_NE(target, nullptr);
  float attacker_reward = 0;
  attacker->init(&attacker_reward);
  float target_reward = 0;
  target->init(&target_reward);

  // Give attacker a laser
  attacker->update_inventory(InventoryItem::laser, 1);
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 1);

  // Give target some items to steal
  target->update_inventory(InventoryItem::heart, 2);
  target->update_inventory(InventoryItem::battery, 3);
  EXPECT_EQ(target->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 3);

  // Assert the attacker is facing the correct direction
  EXPECT_EQ(attacker->orientation, Orientation::Up);

  // Create a grid and add the agents to it. Then set up an attack action handler and call it.
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(attacker);
  grid.add_object(target);

  ActionConfig attack_cfg;
  Attack attack(attack_cfg);
  attack.init(&grid);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(attacker->id, 5, 0);
  EXPECT_TRUE(success);

  // Verify laser was consumed
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 0);

  // Verify target was frozen
  EXPECT_GT(target->frozen, 0);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory[InventoryItem::heart], 0);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 0);
  EXPECT_EQ(attacker->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(attacker->inventory[InventoryItem::battery], 3);

  // grid will delete the agents
}
