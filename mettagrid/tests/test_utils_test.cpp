// test_utils_test.cpp
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <iostream>

#include "core.hpp"
#include "grid.hpp"

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

// Test creating action array
TEST_F(TestUtilsTest, CreateActionArray) {
  std::cout << "Creating action array..." << std::endl;
  uint32_t num_agents = 2;
  int32_t** actions = test_utils::create_action_array(num_agents);
  std::cout << "Action array created." << std::endl;

  ASSERT_NE(actions, nullptr) << "Actions array should not be null";

  // Test action values
  EXPECT_EQ(actions[0][0], 0) << "Default action type should be 0";
  EXPECT_EQ(actions[0][1], 0) << "Default action arg should be 0";

  // Clean up
  test_utils::delete_action_array(actions, num_agents);
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
  std::cout << "Creating grid for manual buffer allocation..." << std::endl;
  auto grid = test_utils::create_test_grid();
  std::cout << "Grid created for manual buffer allocation." << std::endl;

  // Manually allocate buffers
  std::cout << "Allocating buffers manually..." << std::endl;
  size_t obs_size = grid->get_observations_size();
  size_t terminals_size = grid->get_terminals_size();
  size_t truncations_size = grid->get_truncations_size();
  size_t rewards_size = grid->get_rewards_size();
  size_t episode_rewards_size = grid->get_episode_rewards_size();
  size_t group_rewards_size = grid->get_group_rewards_size();

  std::cout << "Buffer sizes calculated." << std::endl;

  // Check that sizes make sense
  EXPECT_GT(obs_size, 0u) << "Observations size should be positive";
  EXPECT_EQ(terminals_size, 2u) << "Terminals size should match num_agents";
  EXPECT_EQ(truncations_size, 2u) << "Truncations size should match num_agents";
  EXPECT_EQ(rewards_size, 2u) << "Rewards size should match num_agents";
  EXPECT_EQ(episode_rewards_size, 2u) << "Episode rewards size should match num_agents";

  std::cout << "Allocating memory..." << std::endl;
  ObsType* observations = new ObsType[obs_size]();
  int8_t* terminals = new int8_t[terminals_size]();
  int8_t* truncations = new int8_t[truncations_size]();
  float* rewards = new float[rewards_size]();
  float* episode_rewards = new float[episode_rewards_size]();
  float* group_rewards = new float[group_rewards_size > 0 ? group_rewards_size : 1]();

  std::cout << "Memory allocated." << std::endl;

  // Set the buffers
  std::cout << "Setting buffers..." << std::endl;
  grid->set_buffers(observations, terminals, truncations, rewards, episode_rewards, group_rewards);
  std::cout << "Buffers set." << std::endl;

  // Try to reset the grid
  std::cout << "Resetting grid..." << std::endl;
  grid->reset();
  std::cout << "Grid reset complete." << std::endl;

  // Try to step the grid
  std::cout << "Creating actions for step..." << std::endl;
  int32_t** actions = test_utils::create_action_array(2);
  std::cout << "Stepping grid..." << std::endl;
  grid->step(actions);
  std::cout << "Grid step complete." << std::endl;

  // Clean up
  std::cout << "Cleaning up..." << std::endl;
  test_utils::delete_action_array(actions, 2);

  delete[] observations;
  delete[] terminals;
  delete[] truncations;
  delete[] rewards;
  delete[] episode_rewards;
  delete[] group_rewards;

  std::cout << "Manual buffer allocation test passed." << std::endl;
}

// Test buffer helpers (once they're implemented)
TEST_F(TestUtilsTest, BufferHelpers) {
  std::cout << "Creating grid for buffer helper test..." << std::endl;

  // Create a struct to hold our buffer functions
  struct BufferHelpers {
    // Function to allocate buffers
    static void allocate_buffers(CppMettaGrid* grid,
                                 ObsType** observations,
                                 int8_t** terminals,
                                 int8_t** truncations,
                                 float** rewards,
                                 float** episode_rewards,
                                 float** group_rewards) {
      // Get buffer sizes
      size_t obs_size = grid->get_observations_size();
      size_t terminals_size = grid->get_terminals_size();
      size_t truncations_size = grid->get_truncations_size();
      size_t rewards_size = grid->get_rewards_size();
      size_t episode_rewards_size = grid->get_episode_rewards_size();
      size_t group_rewards_size = grid->get_group_rewards_size();

      // Allocate buffers
      *observations = new ObsType[obs_size]();
      *terminals = new int8_t[terminals_size]();
      *truncations = new int8_t[truncations_size]();
      *rewards = new float[rewards_size]();
      *episode_rewards = new float[episode_rewards_size]();
      *group_rewards = new float[group_rewards_size > 0 ? group_rewards_size : 1]();

      // Connect buffers to grid
      grid->set_buffers(*observations, *terminals, *truncations, *rewards, *episode_rewards, *group_rewards);
    }

    // Function to free buffers
    static void free_buffers(ObsType* observations,
                             int8_t* terminals,
                             int8_t* truncations,
                             float* rewards,
                             float* episode_rewards,
                             float* group_rewards) {
      delete[] observations;
      delete[] terminals;
      delete[] truncations;
      delete[] rewards;
      delete[] episode_rewards;
      delete[] group_rewards;
    }
  };

  // Create grid
  auto grid = test_utils::create_test_grid();

  // Pointers for our buffers
  ObsType* observations = nullptr;
  int8_t* terminals = nullptr;
  int8_t* truncations = nullptr;
  float* rewards = nullptr;
  float* episode_rewards = nullptr;
  float* group_rewards = nullptr;

  // Allocate and connect buffers
  std::cout << "Allocating buffers with helper..." << std::endl;
  BufferHelpers::allocate_buffers(
      grid.get(), &observations, &terminals, &truncations, &rewards, &episode_rewards, &group_rewards);
  std::cout << "Buffers allocated with helper." << std::endl;

  // Try grid operations
  std::cout << "Resetting grid..." << std::endl;
  grid->reset();
  std::cout << "Grid reset." << std::endl;

  std::cout << "Creating actions..." << std::endl;
  int32_t** actions = test_utils::create_action_array(2);
  std::cout << "Stepping grid..." << std::endl;
  grid->step(actions);
  std::cout << "Grid step complete." << std::endl;

  // Clean up
  test_utils::delete_action_array(actions, 2);

  std::cout << "Freeing buffers with helper..." << std::endl;
  BufferHelpers::free_buffers(observations, terminals, truncations, rewards, episode_rewards, group_rewards);
  std::cout << "Buffers freed with helper." << std::endl;

  std::cout << "Buffer helper test passed." << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Turn on verbose logging
  testing::FLAGS_gtest_break_on_failure = true;
  return RUN_ALL_TESTS();
}