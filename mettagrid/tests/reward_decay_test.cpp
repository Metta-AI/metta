#include <gtest/gtest.h>

#include "actions/noop.hpp"
#include "core.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "stats_tracker.hpp"
#include "test_utils.hpp"

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

    // Create action array for testing
    actions = test_utils::create_action_array(kNumAgents);
  }

  void TearDown() override {
    // Clean up actions array
    test_utils::delete_action_array(actions, kNumAgents);
  }

  // Helper method to run steps and compare reward ratios
  void run_steps_and_check_ratio(float expected_ratio, int setup_steps, int test_steps) {
    // Run setup steps first
    for (int step = 0; step < setup_steps; ++step) {
      metta_grid->step(actions);
    }

    // Store the rewards at this point
    auto& initial_rewards = metta_grid->get_rewards();
    std::vector<float> stored_rewards(initial_rewards.begin(), initial_rewards.end());

    // Run test steps
    for (int i = 0; i < test_steps; ++i) {
      metta_grid->step(actions);
    }

    // Get final rewards for comparison
    auto& final_rewards = metta_grid->get_rewards();

    // Compare ratios where relevant
    for (uint32_t i = 0; i < kNumAgents; ++i) {
      if (fabs(stored_rewards[i]) > 0.001f) {
        float ratio = final_rewards[i] / stored_rewards[i];
        EXPECT_NEAR(expected_ratio, ratio, 0.001f) << "Expected ratio of " << expected_ratio << ", got " << ratio;
      }
    }
  }

  std::unique_ptr<MettaGrid> metta_grid;
  int32_t** actions;
};

// Test minimum reward decay limit
TEST_F(RewardDecayTest, MinimumDecayLimit) {
  // Reset and enable decay with a small time constant
  metta_grid->reset();
  metta_grid->enable_reward_decay(5);  // Fast decay

  // Calculate how many steps needed to reach minimum
  const float kDecayFactor = 3.0f / 5;
  float multiplier = 1.0f;
  int steps_to_minimum = 0;

  while (multiplier > 0.1f) {
    multiplier *= (1.0f - kDecayFactor);
    steps_to_minimum++;
  }

  // Add a few extra steps to be sure
  int setup_steps = steps_to_minimum + 5;

  // Run steps and check the minimum ratio is applied
  run_steps_and_check_ratio(0.1f, setup_steps, 5);
}

// Test disabled reward decay
TEST_F(RewardDecayTest, DisabledDecay) {
  // Reset and ensure decay is disabled
  metta_grid->reset();
  metta_grid->disable_reward_decay();

  // With decay disabled, ratio should be 1.0
  run_steps_and_check_ratio(1.0f, 3, 5);
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

  // Disable decay
  metta_grid->disable_reward_decay();

  // Now ratio should be 1.0 since decay is disabled
  run_steps_and_check_ratio(1.0f, 0, 5);
}

// Test changing decay time constant
TEST_F(RewardDecayTest, ChangeDecayTimeConstant) {
  // Reset and enable with slow decay
  metta_grid->reset();
  metta_grid->enable_reward_decay(100);  // Slow decay

  // Run some steps
  for (int i = 0; i < 10; i++) {
    metta_grid->step(actions);
  }

  // Store current rewards
  auto& slow_decay_rewards = metta_grid->get_rewards();
  std::vector<float> stored_slow = {slow_decay_rewards.begin(), slow_decay_rewards.end()};

  // Change to fast decay
  metta_grid->enable_reward_decay(5);  // Fast decay

  // Run same number of steps
  for (int i = 0; i < 10; i++) {
    metta_grid->step(actions);
  }

  // Get rewards after fast decay
  auto& fast_decay_rewards = metta_grid->get_rewards();

  // Fast decay should result in smaller rewards than slow decay
  for (uint32_t i = 0; i < kNumAgents; ++i) {
    if (fabs(stored_slow[i]) > 0.001f && fabs(fast_decay_rewards[i]) > 0.001f) {
      EXPECT_LT(fast_decay_rewards[i], stored_slow[i]) << "Expected fast decay to produce smaller rewards";
    }
  }
}

// Test reward decay calculation precision
TEST_F(RewardDecayTest, DecayPrecision) {
  // Reset and enable with precise decay value
  metta_grid->reset();
  const int32_t kDecayTimeSteps = 50;
  metta_grid->enable_reward_decay(kDecayTimeSteps);

  // Calculate expected decay factor
  const float kDecayFactor = 3.0f / kDecayTimeSteps;

  // Initial multiplier
  float expected_multiplier = 1.0f;

  // Run a moderate number of steps
  const int kStepsToRun = 25;  // Half the decay time

  for (int step = 0; step < kStepsToRun; ++step) {
    // Step the environment
    metta_grid->step(actions);

    // Update expected multiplier
    if (step > 0) {
      expected_multiplier *= (1.0f - kDecayFactor);
      expected_multiplier = std::max(0.1f, expected_multiplier);
    }
  }

  // Store rewards
  auto& rewards_after_steps = metta_grid->get_rewards();

  // Create a fresh grid to compare against
  auto reference_grid =
      test_utils::create_test_grid(kMapWidth, kMapHeight, kNumAgents, kMaxTimestep, kObsWidth, kObsHeight);
  reference_grid->reset();

  // Step once to get baseline rewards
  reference_grid->step(actions);
  auto& reference_rewards = reference_grid->get_rewards();

  // Compare precision of decay - should match our calculated value
  for (uint32_t i = 0; i < kNumAgents; ++i) {
    if (fabs(reference_rewards[i]) > 0.001f && fabs(rewards_after_steps[i]) > 0.001f) {
      float actual_ratio = rewards_after_steps[i] / reference_rewards[i];
      EXPECT_NEAR(expected_multiplier, actual_ratio, 0.001f)
          << "Expected multiplier " << expected_multiplier << ", got " << actual_ratio;
    }
  }
}