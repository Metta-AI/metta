#include <gtest/gtest.h>

#include <memory>

#include "core.hpp"
#include "test_utils.hpp"

// Test fixture for MettaGrid initialization from test files
class MettaGridTestDataTest : public ::testing::Test {
protected:
  std::unique_ptr<CppMettaGrid> grid;

  void SetUp() override {
    // Initialize the grid from test data before each test
    grid = test_utils::create_grid_from_mettagrid_args("tests");
  }

  void TearDown() override {
    // Clean up automatically handled by unique_ptr
  }
};

// Test basic grid initialization
TEST_F(MettaGridTestDataTest, BasicInitialization) {
  ASSERT_TRUE(grid != nullptr);

  // Check map dimensions (these should match your test data)
  EXPECT_EQ(62, grid->map_width());
  EXPECT_EQ(62, grid->map_height());

  // Test if agents are properly loaded
  EXPECT_GT(grid->num_agents(), 0);

  // Test if the grid features are as expected
  auto features = grid->grid_features();
  EXPECT_FALSE(features.empty());
}

// Test agent initialization
TEST_F(MettaGridTestDataTest, AgentInitialization) {
  // Check if agents are present and initialized
  const auto& agents = grid->get_agents();
  EXPECT_FALSE(agents.empty());

  // Count agents of each type/group
  int agent_count = 0;
  for (const auto& agent : agents) {
    // Assuming agents have a group_id() method, adjust as needed
    if (agent != nullptr) {
      agent_count++;
    }
  }

  // Verify agent count matches
  EXPECT_EQ(grid->num_agents(), agent_count);
}

// Test observation generation
TEST_F(MettaGridTestDataTest, ObservationGeneration) {
  // Reset the grid to ensure consistent state
  grid->reset();

  // Create dummy actions - just zeroes
  int32_t** actions = test_utils::create_action_array(grid->num_agents());

  // Step once to generate observations
  grid->step(actions);

  // Check if observations are generated
  const auto* observations = grid->get_observations();
  EXPECT_TRUE(observations != nullptr);
  EXPECT_GT(grid->get_observations_size(), 0);  // Check size is greater than 0

  // Clean up
  test_utils::delete_action_array(actions, grid->num_agents());
}

// Test reward structure
TEST_F(MettaGridTestDataTest, RewardStructure) {
  // Reset the grid
  grid->reset();

  // Check if the reward vectors are initialized
  const auto* rewards = grid->get_rewards();
  EXPECT_TRUE(rewards != nullptr);
  EXPECT_EQ(grid->num_agents(), grid->get_rewards_size());  // Check size matches agent count

  // Check if group rewards are initialized
  const auto* group_rewards = grid->get_group_rewards();
  EXPECT_TRUE(group_rewards != nullptr);
  EXPECT_GT(grid->get_group_rewards_size(), 0);  // Check size is greater than 0
}

// Test object loading
TEST_F(MettaGridTestDataTest, ObjectLoading) {
  // Get JSON representation of grid objects
  std::string objects_json = grid->get_grid_objects_json();

  // Verify it's not empty
  EXPECT_FALSE(objects_json.empty());

  // Parse the JSON to verify object counts
  nlohmann::json objects;
  try {
    objects = nlohmann::json::parse(objects_json);
  } catch (const nlohmann::json::exception& e) {
    FAIL() << "Failed to parse grid objects JSON: " << e.what();
    return;
  }
  // std::cout << "JSON type: " << objects << std::endl;

  // Make sure objects is an object, not an array
  ASSERT_TRUE(objects.is_object()) << "Grid objects JSON is not an object";

  // Check for existence of various object types from the test data
  bool found_wall = false;
  bool found_mine = false;
  bool found_altar = false;

  for (const auto& obj : objects) {
    // Check if "type" field exists and is a string
    if (obj.contains("type") && obj["type"].is_string()) {
      std::string type = obj["type"];
      if (type == "wall")
        found_wall = true;
      else if (type.find("mine") != std::string::npos)
        found_mine = true;
      else if (type == "altar")
        found_altar = true;
    }
    // Alternative approach: look for type_id or any other field that might indicate object type
    else if (obj.contains("type_id")) {
      // This is just an example - adjust based on your actual JSON structure
      int type_id = obj["type_id"];
      // Map type_id to object types based on your game's object mapping
      if (type_id == 1)
        found_wall = true;  // Assuming type_id 1 is wall
      else if (type_id >= 10 && type_id < 20)
        found_mine = true;  // Assuming mines have type_ids in this range
      else if (type_id == 5)
        found_altar = true;  // Assuming type_id 5 is altar
    }
  }

  // Log the actual JSON for debugging
  if (!found_wall || !found_mine || !found_altar) {
    std::cout << "Grid objects JSON: " << objects_json << std::endl;
  }

  EXPECT_TRUE(found_wall) << "Wall object not found in grid";
  EXPECT_TRUE(found_mine) << "Mine object not found in grid";
  EXPECT_TRUE(found_altar) << "Altar object not found in grid";
}

// Test step functionality
TEST_F(MettaGridTestDataTest, StepFunctionality) {
  // Reset the grid
  grid->reset();

  // Check initial timestep
  EXPECT_EQ(0, grid->current_timestep());

  // Create actions - move all agents in direction 0
  int32_t** actions = test_utils::create_action_array(grid->num_agents(), 1, 0);

  // Take a step
  grid->step(actions);

  // Check timestep increment
  EXPECT_EQ(1, grid->current_timestep());

  // Check action success
  auto success = grid->action_success();
  EXPECT_EQ(grid->num_agents(), success.size());

  // Clean up
  test_utils::delete_action_array(actions, grid->num_agents());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}