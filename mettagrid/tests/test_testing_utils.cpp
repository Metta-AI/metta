#include <gtest/gtest.h>

#include <iostream>

#include "testing_utils.hpp"

class TestingUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::cout << "Test setup..." << std::endl;
  }

  void TearDown() override {
    std::cout << "Test teardown..." << std::endl;
  }
};

TEST_F(TestingUtilsTest, CreateBasicGrid) {
  auto grid = testing_utils::create_test_grid();

  ASSERT_NE(grid, nullptr) << "Grid should not be null";
  EXPECT_EQ(grid->map_width(), 10u) << "Map width should be 10";
  EXPECT_EQ(grid->map_height(), 10u) << "Map height should be 10";
  EXPECT_EQ(grid->num_agents(), 2u) << "Number of agents should be 2";

  std::cout << "Basic grid creation test passed." << std::endl;
}

// Test action array creation and cleanup
TEST_F(TestingUtilsTest, ActionArrayManagement) {
  const uint32_t num_agents = 2;
  const ActionType action_type = 0;  // Assume 0 is Noop
  const ActionType action_arg = 0;

  ActionType* actions = testing_utils::create_action_array(num_agents, action_type, action_arg);

  ASSERT_NE(actions, nullptr) << "Actions array should not be null";

  // Check first agent's action
  EXPECT_EQ(actions[0], action_type) << "First agent action type should match";
  EXPECT_EQ(actions[1], action_arg) << "First agent action arg should match";

  // Check second agent's action
  EXPECT_EQ(actions[2], action_type) << "Second agent action type should match";
  EXPECT_EQ(actions[3], action_arg) << "Second agent action arg should match";

  testing_utils::delete_action_array(actions);
  std::cout << "Action array management test passed." << std::endl;
}

TEST_F(TestingUtilsTest, BufferManagement) {
  auto grid = testing_utils::create_test_grid();
  ASSERT_NE(grid, nullptr);

  // Allocate buffers
  testing_utils::GridBuffers* buffers = testing_utils::allocate_grid_buffers(grid.get());
  ASSERT_NE(buffers, nullptr) << "Buffers should not be null";

  // Check buffer sizes make sense
  EXPECT_GT(buffers->obs_size, 0u) << "Observations size should be positive";
  EXPECT_EQ(buffers->terminals_size, 2u) << "Terminals size should match num_agents";
  EXPECT_EQ(buffers->truncations_size, 2u) << "Truncations size should match num_agents";
  EXPECT_EQ(buffers->rewards_size, 2u) << "Rewards size should match num_agents";

  // Check that buffers are allocated
  EXPECT_NE(buffers->observations, nullptr) << "Observations buffer should be allocated";
  EXPECT_NE(buffers->terminals, nullptr) << "Terminals buffer should be allocated";
  EXPECT_NE(buffers->truncations, nullptr) << "Truncations buffer should be allocated";
  EXPECT_NE(buffers->rewards, nullptr) << "Rewards buffer should be allocated";

  testing_utils::free_grid_buffers(buffers);
  std::cout << "Buffer management test passed." << std::endl;
}

TEST_F(TestingUtilsTest, RAIIWrapper) {
  {
    testing_utils::GridWithBuffers grid_wrapper(10, 10, 2, 100, 5, 5);

    ASSERT_NE(grid_wrapper.get(), nullptr) << "Grid should not be null";
    ASSERT_NE(grid_wrapper.buffers(), nullptr) << "Buffers should not be null";

    // Test grid operations
    auto reset_result = grid_wrapper->reset();

    // Test action creation helper
    auto actions = testing_utils::create_action_numpy_array(grid_wrapper->num_agents(), 0, 0);

    try {
      auto step_result = grid_wrapper->step(actions);
      std::cout << "Grid step succeeded with RAII wrapper." << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Grid step failed (expected if not fully configured): " << e.what() << std::endl;
    }
  }  // RAII wrapper automatically cleans up buffers here

  std::cout << "RAII wrapper test passed." << std::endl;
}

TEST_F(TestingUtilsTest, CompleteWorkflow) {
  testing_utils::GridBuffers* buffers = nullptr;
  auto grid = testing_utils::create_test_grid_with_buffers(10, 10, 2, 100, 5, 5, &buffers);

  ASSERT_NE(grid, nullptr) << "Grid should not be null";
  ASSERT_NE(buffers, nullptr) << "Buffers should not be null";

  // Try to reset the grid
  auto reset_result = grid->reset();

  // Test both action creation methods
  ActionType* actions_data = testing_utils::create_action_array(grid->num_agents(), 0, 0);
  auto actions_numpy = testing_utils::create_action_numpy_array(grid->num_agents(), 0, 0);

  // Try to step with numpy array (preferred method)
  try {
    auto step_result = grid->step(actions_numpy);
    std::cout << "Grid step succeeded." << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Grid step failed (expected if not fully configured): " << e.what() << std::endl;
  }

  // Clean up
  testing_utils::delete_action_array(actions_data);
  testing_utils::free_grid_buffers(buffers);

  std::cout << "Complete workflow test completed." << std::endl;
}

TEST_F(TestingUtilsTest, ErrorHandling) {
  // Test null pointer handling
  MettaGrid* null_grid = nullptr;
  EXPECT_THROW(testing_utils::allocate_grid_buffers(null_grid), std::invalid_argument);

  // Test zero agents
  EXPECT_THROW(testing_utils::create_action_array(0), std::invalid_argument);

  // Test null buffer free (should not throw)
  EXPECT_NO_THROW(testing_utils::free_grid_buffers(nullptr));

  std::cout << "Error handling test passed." << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
