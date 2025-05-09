#include <gtest/gtest.h>

#include <memory>

#include "core.hpp"
#include "test_utils.hpp"

// Test fixture for MettaGrid initialization from test files
class MettaGridTestDataTest : public ::testing::Test {
protected:
  std::unique_ptr<CppMettaGrid> grid;

  // External buffers - need to be managed by the test
  ObsType* observations;
  int8_t* terminals;
  int8_t* truncations;
  float* rewards;

  void SetUp() override {
    // Initialize the grid from test data before each test
    grid = test_utils::create_grid_from_mettagrid_args("tests");

    // Allocate external buffers based on grid dimensions
    size_t obs_size = grid->get_observations_size();
    size_t term_size = grid->get_terminals_size();
    size_t trunc_size = grid->get_truncations_size();
    size_t reward_size = grid->get_rewards_size();

    // Create the external buffers with zero initialization
    observations = new ObsType[obs_size]();
    terminals = new int8_t[term_size]();
    truncations = new int8_t[trunc_size]();
    rewards = new float[reward_size]();

    // Set the external buffers in the grid
    grid->set_buffers(observations, terminals, truncations, rewards);
  }

  void TearDown() override {
    // Clean up external buffers
    delete[] observations;
    delete[] terminals;
    delete[] truncations;
    delete[] rewards;

    // Grid cleanup automatically handled by unique_ptr
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
  EXPECT_TRUE(observations != nullptr);
  EXPECT_GT(grid->get_observations_size(), 0);  // Check size is greater than 0

  // Check that the observations buffer has some non-zero data
  bool has_nonzero = false;
  for (size_t i = 0; i < grid->get_observations_size(); i++) {
    if (observations[i] != 0) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Observations buffer should contain some non-zero values after step";

  // Clean up
  test_utils::delete_action_array(actions, grid->num_agents());
}

// Test reward structure
TEST_F(MettaGridTestDataTest, RewardStructure) {
  // Reset the grid
  grid->reset();

  // Check if the reward vectors are initialized
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

  // Make sure objects is an object, not an array
  ASSERT_TRUE(objects.is_object()) << "Grid objects JSON is not an object";

  // Check for existence of various object types from the test data
  bool found_wall = false;
  bool found_mine = false;
  bool found_altar = false;

  // Iterate over key-value pairs in the JSON object
  for (const auto& [key, obj] : objects.items()) {
    // Check if "type_name" field exists
    if (obj.contains("type_name")) {
      std::string type_name = obj["type_name"];
      if (type_name == "wall")
        found_wall = true;
      else if (type_name.find("mine") != std::string::npos)
        found_mine = true;
      else if (type_name == "altar")
        found_altar = true;
    }
  }

  if (false) {
    // Print non-wall object type names for debugging
    std::cout << "Non-wall objects found:" << std::endl;

    // Iterate over key-value pairs in the JSON object
    for (const auto& [key, obj] : objects.items()) {
      // Check if "type_name" field exists
      if (obj.contains("type_name")) {
        std::string type_name = obj["type_name"];

        if (type_name == "wall")
          found_wall = true;
        else if (type_name.find("mine") != std::string::npos)
          found_mine = true;
        else if (type_name == "altar")
          found_altar = true;

        // Print only non-Wall objects to reduce output clutter
        if (type_name != "wall") {
          std::cout << "Object ID " << key << ": Type = " << type_name << std::endl;
          // Print all properties of this object
          std::cout << "  Properties: ";
          for (auto& [prop_name, prop_value] : obj.items()) {
            if (prop_name != "type_name") {
              std::cout << prop_name << "=" << prop_value << ", ";
            }
          }
          std::cout << std::endl;
        }
      }
    }
  }

  EXPECT_TRUE(found_wall) << "Wall object not found in grid";
  EXPECT_TRUE(found_mine) << "Mine object not found in grid";
  EXPECT_TRUE(found_altar) << "Altar object not found in grid";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}