#include <gtest/gtest.h>

#include <iostream>

#include "actions/noop.hpp"
#include "core.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "stats_tracker.hpp"
#include "test_utils.hpp"

// Modify the test class to focus on the multiplier
class RewardDecayTest : public ::testing::Test {
protected:
  // Constants for test setup
  const uint32_t kMapWidth = 10;
  const uint32_t kMapHeight = 10;
  const uint32_t kNumAgents = 2;
  const uint32_t kMaxTimestep = 100;
  const uint16_t kObsWidth = 5;
  const uint16_t kObsHeight = 5;

  void SetUp() override {
    // Create a test grid using the utilities
    metta_grid = test_utils::create_test_grid(kMapWidth, kMapHeight, kNumAgents, kMaxTimestep, kObsWidth, kObsHeight);
    ASSERT_TRUE(metta_grid != nullptr) << "Failed to create test grid";

    // Create action array for testing
    actions = test_utils::create_action_array(kNumAgents);
    ASSERT_TRUE(actions != nullptr) << "Failed to create action array";

    // Initialize the grid to ensure it's in a valid state
    metta_grid->reset();
  }

  void TearDown() override {
    // Clean up actions array
    if (actions != nullptr) {
      test_utils::delete_action_array(actions, kNumAgents);
      actions = nullptr;
    }

    // Reset the grid to ensure clean state
    if (metta_grid != nullptr) {
      metta_grid.reset();
    }
  }

  // Run steps and return the resulting multiplier
  float run_steps_and_get_multiplier(int num_steps) {
    // Run steps
    for (int i = 0; i < num_steps; ++i) {
      metta_grid->step(actions);
    }

    // Return the multiplier directly
    return metta_grid->get_reward_multiplier();
  }

  std::unique_ptr<CppMettaGrid> metta_grid;
  int32_t** actions;
};

// Test minimum reward decay limit
TEST_F(RewardDecayTest, MinimumDecayLimit) {
  // Reset and make sure we have a fresh state
  metta_grid->reset();

  // Enable fast decay
  metta_grid->enable_reward_decay(5);  // Fast decay

  // Run enough steps to ensure we hit the minimum decay limit (0.1)
  // With decay factor of 3.0/5, this should be reached within ~8 steps
  for (int i = 0; i < 20; ++i) {
    metta_grid->step(actions);
  }

  // Check if the multiplier has reached the minimum limit
  float multiplier = metta_grid->get_reward_multiplier();
  EXPECT_NEAR(0.1f, multiplier, 0.001f) << "Expected multiplier to reach minimum 0.1, got " << multiplier;
}

// Test disabled reward decay
TEST_F(RewardDecayTest, DisabledDecay) {
  // Reset and ensure decay is disabled
  metta_grid->reset();
  metta_grid->disable_reward_decay();

  // Run some steps
  float multiplier = run_steps_and_get_multiplier(5);

  // With decay disabled, multiplier should stay at 1.0
  EXPECT_NEAR(1.0f, multiplier, 0.001f) << "Expected multiplier of 1.0 with disabled decay, got " << multiplier;
}

// Test enabling and then disabling decay
TEST_F(RewardDecayTest, EnableThenDisable) {
  // Reset and enable decay
  metta_grid->reset();
  metta_grid->enable_reward_decay(10);

  // Run a few steps to allow some decay
  for (int i = 0; i < 5; i++) {
    metta_grid->step(actions);
  }

  // Get multiplier after decay
  float decayed_multiplier = metta_grid->get_reward_multiplier();
  EXPECT_LT(decayed_multiplier, 1.0f) << "Expected multiplier to decrease from 1.0, got " << decayed_multiplier;

  // Disable decay - this should reset the multiplier to 1.0
  metta_grid->disable_reward_decay();

  // Multiplier should be reset to 1.0 immediately
  float reset_multiplier = metta_grid->get_reward_multiplier();
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
  metta_grid->reset();
  metta_grid->enable_reward_decay(100);  // Slow decay

  // Run some steps with slow decay
  for (int i = 0; i < 10; i++) {
    metta_grid->step(actions);
  }

  // Get multiplier after slow decay
  float slow_multiplier = metta_grid->get_reward_multiplier();

  // Reset and enable with fast decay
  metta_grid->reset();
  metta_grid->enable_reward_decay(5);  // Fast decay

  // Run same number of steps with fast decay
  for (int i = 0; i < 10; i++) {
    metta_grid->step(actions);
  }

  // Get multiplier after fast decay
  float fast_multiplier = metta_grid->get_reward_multiplier();

  // The fast decay should produce a smaller multiplier
  EXPECT_LT(fast_multiplier, slow_multiplier)
      << "Fast decay multiplier (" << fast_multiplier << ") should be smaller than slow decay multiplier ("
      << slow_multiplier << ")";
}

// Test reward decay calculation precision
TEST_F(RewardDecayTest, DecayPrecision) {
  // Reset and enable with precise decay value
  metta_grid->reset();
  const int32_t kDecayTimeSteps = 50;
  metta_grid->enable_reward_decay(kDecayTimeSteps);

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
    metta_grid->step(actions);
  }

  // Get final multiplier
  float actual_multiplier = metta_grid->get_reward_multiplier();

  // Check precision of decay calculation
  EXPECT_NEAR(expected_multiplier, actual_multiplier, 0.001f)
      << "Expected multiplier " << expected_multiplier << ", got " << actual_multiplier;
}