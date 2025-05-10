// test_utils_test.cpp
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <iostream>

#include "core.hpp"
#include "grid.hpp"
#include "types.hpp"

// Test class for examining test_utils functions
class TestUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::cout << "Starting setup..." << std::endl;
  }

  void TearDown() override {
    std::cout << "Finishing teardown..." << std::endl;
  }
};

// Test creating a grid
TEST_F(TestUtilsTest, CreateGrid) {
  std::cout << "Creating grid..." << std::endl;
  auto grid = test_utils::create_test_grid();
  std::cout << "Grid created." << std::endl;

  ASSERT_NE(grid, nullptr) << "Grid should not be null";
  EXPECT_EQ(grid->map_width(), 10u) << "Map width should be 10";
  EXPECT_EQ(grid->map_height(), 10u) << "Map height should be 10";
  EXPECT_EQ(grid->num_agents(), 2u) << "Number of agents should be 2";

  std::cout << "Grid test passed." << std::endl;
}

TEST_F(TestUtilsTest, CreateActionArray) {
  std::cout << "Creating action array..." << std::endl;
  uint32_t num_agents = 2;
  ActionsType* actions = test_utils::create_action_array(num_agents);
  std::cout << "Action array created." << std::endl;

  ASSERT_NE(actions, nullptr) << "Actions array should not be null";

  // Test action values - updated for flat array access
  // First agent's action (idx 0)
  EXPECT_EQ(actions[0], ActionType::Noop) << "Default action type should be Noop";
  EXPECT_EQ(actions[1], 0) << "Noop action arg should be 0";

  // Second agent's action (idx 2)
  EXPECT_EQ(actions[2], ActionType::Noop) << "Default action type should be Noop";
  EXPECT_EQ(actions[3], 0) << "Noop action arg should be 0";

  // Clean up - updated for flat array
  test_utils::delete_action_array(actions);
  std::cout << "Action array test passed." << std::endl;
}

// Test examining grid internals
TEST_F(TestUtilsTest, ExamineGrid) {
  std::cout << "Creating grid for examination..." << std::endl;
  auto grid = test_utils::create_test_grid();
  std::cout << "Grid created for examination." << std::endl;

  // Check buffer sizes
  std::cout << "Examining buffer sizes..." << std::endl;
  std::cout << "Observations size: " << grid->get_observations_size() << std::endl;
  std::cout << "Terminals size: " << grid->get_terminals_size() << std::endl;
  std::cout << "Truncations size: " << grid->get_truncations_size() << std::endl;
  std::cout << "Rewards size: " << grid->get_rewards_size() << std::endl;
  std::cout << "Episode rewards size: " << grid->get_episode_rewards_size() << std::endl;
  std::cout << "Group rewards size: " << grid->get_group_rewards_size() << std::endl;

  EXPECT_GT(grid->get_observations_size(), 0u) << "Observations size should be positive";
  EXPECT_GT(grid->get_terminals_size(), 0u) << "Terminals size should be positive";
  EXPECT_GT(grid->get_truncations_size(), 0u) << "Truncations size should be positive";
  EXPECT_GT(grid->get_rewards_size(), 0u) << "Rewards size should be positive";
  EXPECT_GT(grid->get_episode_rewards_size(), 0u) << "Episode rewards size should be positive";
  // Group rewards might be 0 if no groups configured

  std::cout << "Examining feature sizes..." << std::endl;
  std::cout << "Grid features: " << grid->grid_features().size() << std::endl;
  std::cout << "Observation size from GridObject: " << GridObject::get_observation_size() << std::endl;

  std::cout << "Grid examination test passed." << std::endl;
}

// Test manual buffer allocation
TEST_F(TestUtilsTest, ManualBufferAllocation) {
  try {
    std::cout << "Creating grid for manual buffer allocation..." << std::endl;
    auto grid = test_utils::create_test_grid();
    std::cout << "Grid created for manual buffer allocation." << std::endl;

    // Manually allocate buffers
    std::cout << "Allocating buffers manually..." << std::endl;
    size_t obs_size = grid->get_observations_size();
    size_t terminals_size = grid->get_terminals_size();
    size_t truncations_size = grid->get_truncations_size();
    size_t rewards_size = grid->get_rewards_size();
    std::cout << "Buffer sizes calculated:" << std::endl;
    std::cout << "  Observations size: " << obs_size << std::endl;
    std::cout << "  Terminals size: " << terminals_size << std::endl;
    std::cout << "  Truncations size: " << truncations_size << std::endl;
    std::cout << "  Rewards size: " << rewards_size << std::endl;

    // Check that sizes make sense
    ASSERT_GT(obs_size, 0u) << "Observations size should be positive";
    ASSERT_EQ(terminals_size, 2u) << "Terminals size should match num_agents";
    ASSERT_EQ(truncations_size, 2u) << "Truncations size should match num_agents";
    ASSERT_EQ(rewards_size, 2u) << "Rewards size should match num_agents";

    std::cout << "Allocating memory..." << std::endl;
    ObsType* observations = new ObsType[obs_size]();
    numpy_bool_t* terminals = new numpy_bool_t[terminals_size]();
    numpy_bool_t* truncations = new numpy_bool_t[truncations_size]();
    float* rewards = new float[rewards_size]();
    std::cout << "Memory allocated." << std::endl;

    // Set the buffers
    std::cout << "Setting buffers..." << std::endl;
    grid->set_buffers(observations, terminals, truncations, rewards);
    std::cout << "Buffers set." << std::endl;

    // Try to reset the grid
    std::cout << "Resetting grid..." << std::endl;
    grid->reset();
    std::cout << "Grid reset complete." << std::endl;

    // Try to step the grid with a valid action
    std::cout << "Creating actions for step..." << std::endl;
    // Make sure the action handler is initialized correctly
    auto action_names = grid->action_names();
    std::cout << "Available actions:" << std::endl;
    for (size_t i = 0; i < action_names.size(); i++) {
      std::cout << "  [" << i << "] " << action_names[i] << std::endl;
    }

    // Find the index for Noop action
    bool can_step = false;
    int action_idx = ActionType::Noop;  // Default to Noop action which is now defined by enum

    if (!action_names.empty()) {
      can_step = true;
      std::cout << "Using action index " << action_idx << " ("
                << (action_idx < action_names.size() ? action_names[action_idx] : "unknown") << ") with max arg "
                << static_cast<int>(ActionMaxArgs[action_idx]) << std::endl;
    } else {
      std::cout << "No actions available, skipping step test" << std::endl;
    }

    if (can_step) {
      // Create a flat action array instead of an array of pointers
      ActionsType* flat_actions = test_utils::create_action_array(grid->num_agents(), action_idx, 0);

      std::cout << "Stepping grid..." << std::endl;
      // Call step with the flat array
      grid->step(flat_actions);
      std::cout << "Grid step complete." << std::endl;

      // Check if rewards were updated
      std::cout << "Checking rewards after step:" << std::endl;
      for (size_t i = 0; i < rewards_size; i++) {
        std::cout << " Reward " << i << ": " << rewards[i] << std::endl;
      }

      // Clean up the flat array - no need for num_agents parameter
      test_utils::delete_action_array(flat_actions);
    }

    // Clean up
    std::cout << "Cleaning up..." << std::endl;
    delete[] observations;
    delete[] terminals;
    delete[] truncations;
    delete[] rewards;

    std::cout << "Manual buffer allocation test passed." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    FAIL() << "Exception occurred during test: " << e.what();
  }
}

// Test buffer helpers (once they're implemented)
TEST_F(TestUtilsTest, BufferHelpers) {
  try {
    std::cout << "Creating grid for buffer helper test..." << std::endl;

    // Create a struct to hold our buffer functions
    struct BufferHelpers {
      // Function to allocate buffers
      static void allocate_buffers(CppMettaGrid* grid,
                                   ObsType** observations,
                                   numpy_bool_t** terminals,
                                   numpy_bool_t** truncations,
                                   float** rewards) {
        // Get buffer sizes
        size_t obs_size = grid->get_observations_size();
        size_t terminals_size = grid->get_terminals_size();
        size_t truncations_size = grid->get_truncations_size();
        size_t rewards_size = grid->get_rewards_size();

        // Allocate buffers
        *observations = new ObsType[obs_size]();
        *terminals = new numpy_bool_t[terminals_size]();
        *truncations = new numpy_bool_t[truncations_size]();
        *rewards = new float[rewards_size]();

        // Connect buffers to grid
        grid->set_buffers(*observations, *terminals, *truncations, *rewards);
      }

      // Function to free buffers
      static void free_buffers(ObsType* observations,
                               numpy_bool_t* terminals,
                               numpy_bool_t* truncations,
                               float* rewards) {
        delete[] observations;
        delete[] terminals;
        delete[] truncations;
        delete[] rewards;
      }
    };

    // Create grid
    auto grid = test_utils::create_test_grid();

    // Pointers for our buffers
    ObsType* observations = nullptr;
    numpy_bool_t* terminals = nullptr;
    numpy_bool_t* truncations = nullptr;
    float* rewards = nullptr;

    // Allocate and connect buffers
    std::cout << "Allocating buffers with helper..." << std::endl;
    BufferHelpers::allocate_buffers(grid.get(), &observations, &terminals, &truncations, &rewards);
    std::cout << "Buffers allocated with helper." << std::endl;

    // Try grid operations
    std::cout << "Resetting grid..." << std::endl;
    grid->reset();
    std::cout << "Grid reset." << std::endl;

    // Find the Noop action or use the first available action
    auto action_names = grid->action_names();
    std::cout << "Available actions:" << std::endl;
    for (size_t i = 0; i < action_names.size(); i++) {
      std::cout << "  [" << i << "] " << action_names[i] << std::endl;
    }

    bool can_step = false;
    int action_idx = 0;

    if (!action_names.empty()) {
      can_step = true;
      // Find the Noop action if available
      for (size_t i = 0; i < action_names.size(); i++) {
        if (action_names[i] == "noop") {
          action_idx = i;
          break;
        }
      }
      std::cout << "Using action index " << action_idx << " ("
                << (action_idx < action_names.size() ? action_names[action_idx] : "unknown") << ")" << std::endl;
    } else {
      std::cout << "No actions available, skipping step test" << std::endl;
    }

    if (can_step) {
      std::cout << "Creating actions..." << std::endl;
      // Create a flat action array
      ActionsType* flat_actions = test_utils::create_action_array(grid->num_agents(), action_idx, 0);

      std::cout << "Stepping grid..." << std::endl;
      // Pass the flat array to step
      grid->step(flat_actions);
      std::cout << "Grid step complete." << std::endl;

      // Clean up with the simpler delete function
      test_utils::delete_action_array(flat_actions);
    }

    // Clean up buffers regardless of whether step succeeded
    std::cout << "Freeing buffers with helper..." << std::endl;
    BufferHelpers::free_buffers(observations, terminals, truncations, rewards);
    std::cout << "Buffers freed with helper." << std::endl;

    std::cout << "Buffer helper test passed." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    FAIL() << "Exception occurred during test: " << e.what();
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Turn on verbose logging
  testing::FLAGS_gtest_break_on_failure = true;
  return RUN_ALL_TESTS();
}