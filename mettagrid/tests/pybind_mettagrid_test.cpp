#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

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
  EXPECT_EQ(agent->max_items, 123);        // Agent config preserved

  // Verify merged rewards
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::heart], 1.0);      // Agent reward preserved
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_red], 0.0);    // Group reward overrides
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_green], 0.5);  // Group reward added

  delete agent;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
