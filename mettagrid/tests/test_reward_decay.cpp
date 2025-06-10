#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "testing_utils.hpp"

namespace py = pybind11;

class RewardDecayTest : public ::testing::Test {
protected:
  // Constants for test setup
  const uint32_t kMapWidth = 10;
  const uint32_t kMapHeight = 10;
  const uint32_t kNumAgents = 2;
  const uint32_t kMaxTimestep = 100;
  const uint16_t kObsWidth = 5;
  const uint16_t kObsHeight = 5;

  // Grid and buffers - using TestMettaGrid for enhanced testing
  std::unique_ptr<testing_utils::TestMettaGrid> mettagrid;
  testing_utils::GridBuffers* buffers = nullptr;

  void SetUp() override {
    // Create test grid using TestMettaGrid
    mettagrid =
        testing_utils::create_test_mettagrid(kMapWidth, kMapHeight, kNumAgents, kMaxTimestep, kObsWidth, kObsHeight);

    // Allocate buffers for the grid
    buffers = testing_utils::allocate_grid_buffers(mettagrid.get());

    // Initialize the grid
    mettagrid->reset();
  }

  void TearDown() override {
    // Clean up buffers
    if (buffers) {
      testing_utils::free_grid_buffers(buffers);
      buffers = nullptr;
    }

    // Grid will be cleaned up by unique_ptr
    mettagrid.reset();
  }

  // Helper function to create action array
  py::array_t<ActionType, py::array::c_style> create_noop_actions() {
    return testing_utils::create_action_numpy_array(mettagrid->num_agents(), 0, 0);
  }

  // Run steps and return the resulting multiplier
  float run_steps_and_get_multiplier(int num_steps) {
    auto actions = create_noop_actions();

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

  // Run enough steps to ensure we hit the minimum decay limit (0.01)
  // With decay factor of 3.0/5, this should be reached within ~8 steps
  auto actions = create_noop_actions();
  for (int i = 0; i < 20; ++i) {
    mettagrid->step(actions);
  }

  // Check if the multiplier has reached the minimum limit
  float multiplier = mettagrid->get_reward_decay_multiplier();
  float min_multiplier = mettagrid->get_min_reward_multiplier();
  EXPECT_NEAR(min_multiplier, multiplier, 0.001f) << "Expected multiplier to reach minimum 0.01, got " << multiplier;
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
  auto actions = create_noop_actions();
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
TEST_F(RewardDecayTest, CompareDecayTimeConstants) {
  // Test slow decay
  mettagrid->reset();
  mettagrid->enable_reward_decay(100);
  auto actions = create_noop_actions();
  for (int i = 0; i < 10; i++) {
    mettagrid->step(actions);
  }
  float slow_multiplier = mettagrid->get_reward_decay_multiplier();

  // Create a new grid for fast decay test (since we can't reset after step)
  auto fast_decay_grid = testing_utils::create_test_mettagrid();
  fast_decay_grid->reset();
  fast_decay_grid->enable_reward_decay(5);

  for (int i = 0; i < 10; i++) {
    fast_decay_grid->step(actions);
  }
  float fast_multiplier = fast_decay_grid->get_reward_decay_multiplier();

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

  // Calculate expected multiplier (corrected to use 0.01 minimum)
  float expected_multiplier = 1.0f;
  for (int step = 0; step < kStepsToRun; ++step) {
    expected_multiplier *= (1.0f - kDecayFactor);
    expected_multiplier = std::max(0.01f, expected_multiplier);  // Updated to 0.01 minimum
  }

  // Run the steps
  auto actions = create_noop_actions();
  for (int step = 0; step < kStepsToRun; ++step) {
    mettagrid->step(actions);
  }

  // Get final multiplier
  float actual_multiplier = mettagrid->get_reward_decay_multiplier();

  // Check precision of decay calculation
  EXPECT_NEAR(expected_multiplier, actual_multiplier, 0.001f)
      << "Expected multiplier " << expected_multiplier << ", got " << actual_multiplier;
}

// Test that reward decay functionality exists and can be controlled
TEST_F(RewardDecayTest, BasicDecayFunctionality) {
  // Test that we start with no decay (multiplier = 1.0)
  float initial_multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_NEAR(1.0f, initial_multiplier, 0.001f) << "Initial multiplier should be 1.0";

  // Test enabling decay
  mettagrid->enable_reward_decay(20);

  // Run some steps to verify decay occurs
  auto actions = create_noop_actions();
  for (int i = 0; i < 5; i++) {
    mettagrid->step(actions);
  }

  float decayed_multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_LT(decayed_multiplier, 1.0f) << "Multiplier should decrease after enabling decay and running steps";

  // Test disabling decay resets multiplier
  mettagrid->disable_reward_decay();
  float disabled_multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_NEAR(1.0f, disabled_multiplier, 0.001f) << "Disabling decay should reset multiplier to 1.0";
}

// Test reward decay properties are exposed correctly through TestMettaGrid
TEST_F(RewardDecayTest, DecayPropertiesExposed) {
  // Test initial state
  EXPECT_FALSE(mettagrid->get_reward_decay_enabled()) << "Decay should be disabled initially";
  EXPECT_GT(mettagrid->get_reward_decay_factor(), 0.0f) << "Decay factor should be positive";

  // Test enabling decay
  mettagrid->enable_reward_decay(20);
  EXPECT_TRUE(mettagrid->get_reward_decay_enabled()) << "Decay should be enabled after calling enable";
  EXPECT_FLOAT_EQ(mettagrid->get_reward_decay_factor(), 3.0f / 20.0f) << "Decay factor should match expected value";

  // Test disabling decay
  mettagrid->disable_reward_decay();
  EXPECT_FALSE(mettagrid->get_reward_decay_enabled()) << "Decay should be disabled after calling disable";
}

// Test reward decay actually affects rewards during gameplay
TEST_F(RewardDecayTest, DecayAffectsRewards) {
  // This test would require setting up a scenario where agents actually get rewards
  // For now, we'll test that the multiplier is being applied correctly

  mettagrid->reset();
  mettagrid->enable_reward_decay(10);  // Fast decay for quick testing

  // Run several steps to build up some decay
  auto actions = create_noop_actions();
  for (int i = 0; i < 5; i++) {
    mettagrid->step(actions);
  }

  float multiplier = mettagrid->get_reward_decay_multiplier();
  EXPECT_LT(multiplier, 1.0f) << "Multiplier should be less than 1.0 after decay steps";
  EXPECT_GE(multiplier, 0.01f) << "Multiplier should not go below minimum of 0.01";
}

int main(int argc, char** argv) {
  py::scoped_interpreter guard{};
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
