#include <gtest/gtest.h>

#include <iostream>

#include "actions/noop.hpp"
#include "core.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "stats_tracker.hpp"
#include "test_utils.hpp"
#include "types.hpp"

class RewardDecayTest : public ::testing::Test {
protected:
  // Constants for test setup
  const uint32_t kMapWidth = 10;
  const uint32_t kMapHeight = 10;
  const uint32_t kNumAgents = 2;
  const uint32_t kMaxTimestep = 100;
  const uint16_t kObsWidth = 5;
  const uint16_t kObsHeight = 5;

  // Grid and action pointers
  std::unique_ptr<CppMettaGrid> mettagrid;
  c_actions_type* actions = nullptr;

  // Grid buffers management
  test_utils::GridBuffers* buffers = nullptr;

  void SetUp() override {
    // Create test grid
    mettagrid = test_utils::create_test_grid(kMapWidth, kMapHeight, kNumAgents, kMaxTimestep, kObsWidth, kObsHeight);

    // Allocate buffers for the grid
    buffers = test_utils::allocate_grid_buffers(mettagrid.get());

    // Create action array for testing
    actions = test_utils::create_action_array(kNumAgents);

    // Initialize the grid
    mettagrid->reset();
  }

  void TearDown() override {
    // Clean up actions array
    if (actions) {
      test_utils::delete_action_array(actions);
      actions = nullptr;
    }

    // Free buffers
    if (buffers) {
      test_utils::free_grid_buffers(buffers);
      buffers = nullptr;
    }

    // Grid will be cleaned up by unique_ptr
    mettagrid.reset();
  }

  // Run steps and return the resulting multiplier
  float run_steps_and_get_multiplier(int num_steps) {
    // Run steps
    for (int i = 0; i < num_steps; ++i) {
      mettagrid->step(actions);
    }

    // Return the multiplier directly
    return mettagrid->get_reward_decay_multiplier();
  }
};

// Test minimum reward decay limit
TEST_F(RewardDecayTest, MinimumDecayLimit) {
  // Reset and make sure we have a fresh state
  mettagrid->reset();

  // Enable fast decay
  mettagrid->enable_reward_decay(5);  // Fast decay

  // Run enough steps to ensure we hit the minimum decay limit (0.1)
  // With decay factor of 3.0/5, this should be reached within ~8 steps
  for (int i = 0; i < 20; ++i) {
    mettagrid->step(actions);
  }

  // Check if the multiplier has reached the minimum limit
  float multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_NEAR(0.1f, multiplier, 0.001f) << "Expected multiplier to reach minimum 0.1, got " << multiplier;
}

// Test disabled reward decay
TEST_F(RewardDecayTest, DisabledDecay) {
  // Reset and ensure decay is disabled
  mettagrid->reset();
  mettagrid->disable_reward_decay();

  // Run some steps
  float multiplier = run_steps_and_get_multiplier(5);

  // With decay disabled, multiplier should stay at 1.0
  EXPECT_NEAR(1.0f, multiplier, 0.001f) << "Expected multiplier of 1.0 with disabled decay, got " << multiplier;
}

// Test enabling and then disabling decay
TEST_F(RewardDecayTest, EnableThenDisable) {
  // Reset and enable decay
  mettagrid->reset();
  mettagrid->enable_reward_decay(10);

  // Run a few steps to allow some decay
  for (int i = 0; i < 5; i++) {
    mettagrid->step(actions);
  }

  // Get multiplier after decay
  float decayed_multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_LT(decayed_multiplier, 1.0f) << "Expected multiplier to decrease from 1.0, got " << decayed_multiplier;

  // Disable decay - this should reset the multiplier to 1.0
  mettagrid->disable_reward_decay();

  // Multiplier should be reset to 1.0 immediately
  float reset_multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_NEAR(1.0f, reset_multiplier, 0.001f)
      << "Expected multiplier to reset to 1.0 after disabling decay, got " << reset_multiplier;

  // Run a few more steps to ensure it stays at 1.0
  float final_multiplier = run_steps_and_get_multiplier(5);
  EXPECT_NEAR(1.0f, final_multiplier, 0.001f)
      << "Expected multiplier to remain at 1.0 after running steps with decay disabled, got " << final_multiplier;
}

// Test changing decay time constant
TEST_F(RewardDecayTest, ChangeDecayTimeConstant) {
  // Reset and enable with slow decay
  mettagrid->reset();
  mettagrid->enable_reward_decay(100);  // Slow decay

  // Run some steps with slow decay
  for (int i = 0; i < 10; i++) {
    mettagrid->step(actions);
  }

  // Get multiplier after slow decay
  float slow_multiplier = mettagrid->get_reward_decay_multiplier();

  // Reset and enable with fast decay
  mettagrid->reset();
  mettagrid->enable_reward_decay(5);  // Fast decay

  // Run same number of steps with fast decay
  for (int i = 0; i < 10; i++) {
    mettagrid->step(actions);
  }

  // Get multiplier after fast decay
  float fast_multiplier = mettagrid->get_reward_decay_multiplier();

  // The fast decay should produce a smaller multiplier
  EXPECT_LT(fast_multiplier, slow_multiplier)
      << "Fast decay multiplier (" << fast_multiplier << ") should be smaller than slow decay multiplier ("
      << slow_multiplier << ")";
}

// Test reward decay calculation precision
TEST_F(RewardDecayTest, DecayPrecision) {
  // Reset and enable with precise decay value
  mettagrid->reset();
  const int32_t kDecayTimeSteps = 50;
  mettagrid->enable_reward_decay(kDecayTimeSteps);

  // Calculate expected decay factor
  const float kDecayFactor = 3.0f / kDecayTimeSteps;

  // Run a moderate number of steps
  const int kStepsToRun = 25;  // Half the decay time

  // Calculate expected multiplier
  float expected_multiplier = 1.0f;
  for (int step = 0; step < kStepsToRun; ++step) {
    expected_multiplier *= (1.0f - kDecayFactor);
    expected_multiplier = std::max(0.1f, expected_multiplier);
  }

  // Run the steps
  for (int step = 0; step < kStepsToRun; ++step) {
    mettagrid->step(actions);
  }

  // Get final multiplier
  float actual_multiplier = mettagrid->get_reward_decay_multiplier();

  // Check precision of decay calculation
  EXPECT_NEAR(expected_multiplier, actual_multiplier, 0.001f)
      << "Expected multiplier " << expected_multiplier << ", got " << actual_multiplier;
}