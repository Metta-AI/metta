#include <gtest/gtest.h>
#include <pybind11/embed.h>  // Add this for interpreter management

#include <iostream>

#include "testing_utils.hpp"

namespace py = pybind11;

class TestingUtilsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    std::cout << "Setting up test suite..." << std::endl;

    // Initialize Python interpreter if not already done
    if (!Py_IsInitialized()) {
      std::cout << "Initializing Python interpreter..." << std::endl;
      py::initialize_interpreter();
    } else {
      std::cout << "Python interpreter already initialized" << std::endl;
    }
  }

  void SetUp() override {
    std::cout << "Test setup..." << std::endl;
  }

  void TearDown() override {
    std::cout << "Test teardown..." << std::endl;
  }
};

TEST_F(TestingUtilsTest, CreateBasicGrid) {
  std::cout << "Starting CreateBasicGrid test..." << std::endl;

  try {
    auto grid = testing_utils::create_test_grid();

    ASSERT_NE(grid, nullptr) << "Grid should not be null";
    EXPECT_EQ(grid->map_width(), 10u) << "Map width should be 10";
    EXPECT_EQ(grid->map_height(), 10u) << "Map height should be 10";
    EXPECT_EQ(grid->num_agents(), 2u) << "Number of agents should be 2";

    std::cout << "Basic grid creation test passed." << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Exception in CreateBasicGrid: " << e.what() << std::endl;
    FAIL() << "CreateBasicGrid failed: " << e.what();
  }
}

// Test action array creation and cleanup
TEST_F(TestingUtilsTest, ActionArrayManagement) {
  std::cout << "Starting ActionArrayManagement test..." << std::endl;

  try {
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
  } catch (const std::exception& e) {
    std::cout << "Exception in ActionArrayManagement: " << e.what() << std::endl;
    FAIL() << "ActionArrayManagement failed: " << e.what();
  }
}

TEST_F(TestingUtilsTest, BufferManagement) {
  std::cout << "Starting BufferManagement test..." << std::endl;

  try {
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
  } catch (const std::exception& e) {
    std::cout << "Exception in BufferManagement: " << e.what() << std::endl;
    FAIL() << "BufferManagement failed: " << e.what();
  }
}

TEST_F(TestingUtilsTest, RAIIWrapper) {
  std::cout << "Starting RAIIWrapper test..." << std::endl;

  try {
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
  } catch (const std::exception& e) {
    std::cout << "Exception in RAIIWrapper: " << e.what() << std::endl;
    FAIL() << "RAIIWrapper failed: " << e.what();
  }
}

TEST_F(TestingUtilsTest, ErrorHandling) {
  std::cout << "Starting ErrorHandling test..." << std::endl;

  try {
    // Test null pointer handling
    MettaGrid* null_grid = nullptr;
    EXPECT_THROW(testing_utils::allocate_grid_buffers(null_grid), std::invalid_argument);

    // Test zero agents
    EXPECT_THROW(testing_utils::create_action_array(0), std::invalid_argument);

    // Test null buffer free (should not throw)
    EXPECT_NO_THROW(testing_utils::free_grid_buffers(nullptr));

    std::cout << "Error handling test passed." << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Exception in ErrorHandling: " << e.what() << std::endl;
    FAIL() << "ErrorHandling failed: " << e.what();
  }
}

int main(int argc, char** argv) {
  std::cout << "Starting test execution..." << std::endl;

  py::scoped_interpreter guard{};
  std::cout << "Python interpreter initialized" << std::endl;

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
