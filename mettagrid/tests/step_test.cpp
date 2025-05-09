#include <gtest/gtest.h>

#include <cassert>
#include <iostream>

#include "core.hpp"
#include "grid.hpp"
#include "test_utils.hpp"
#include "types.hpp"

// Single test function that performs a minimal grid step
TEST(StepTest, BasicStep) {
  try {
    std::cout << "======= Starting Minimal Step Test =======" << std::endl;

    // Create grid
    std::cout << "Creating grid..." << std::endl;
    auto grid = test_utils::create_test_grid();
    assert(grid != nullptr);
    std::cout << "Grid created." << std::endl;

    // Print grid dimensions
    std::cout << "Grid dimensions: " << grid->map_width() << "x" << grid->map_height() << std::endl;
    std::cout << "Number of agents: " << grid->num_agents() << std::endl;

    // Allocate buffers using minimal sizes directly
    std::cout << "Allocating minimal buffers..." << std::endl;
    size_t obs_size = grid->get_observations_size();
    size_t term_size = grid->get_terminals_size();
    size_t trunc_size = grid->get_truncations_size();
    size_t reward_size = grid->get_rewards_size();

    std::cout << "Buffer sizes: " << std::endl;
    std::cout << "  Observations: " << obs_size << std::endl;
    std::cout << "  Terminals: " << term_size << std::endl;
    std::cout << "  Truncations: " << trunc_size << std::endl;
    std::cout << "  Rewards: " << reward_size << std::endl;
    std::cout << "  Episode rewards: Internal (managed by grid)" << std::endl;
    std::cout << "  Group rewards: Internal (managed by grid)" << std::endl;

    // Create the external buffers with zero initialization
    ObsType* observations = new ObsType[obs_size]();
    int8_t* terminals = new int8_t[term_size]();
    int8_t* truncations = new int8_t[trunc_size]();
    float* rewards = new float[reward_size]();

    std::cout << "Buffers allocated." << std::endl;

    // Set the buffers
    std::cout << "Setting buffers..." << std::endl;
    grid->set_buffers(observations, terminals, truncations, rewards);
    std::cout << "Buffers set." << std::endl;

    // Reset the grid
    std::cout << "Resetting grid..." << std::endl;
    grid->reset();
    std::cout << "Grid reset." << std::endl;

    // Get action names
    std::cout << "Available actions:" << std::endl;
    auto action_names = grid->action_names();
    for (size_t i = 0; i < action_names.size(); i++) {
      std::cout << "  [" << i << "] " << action_names[i] << std::endl;
    }

    // Check if we have a Noop action
    int noop_idx = -1;
    for (size_t i = 0; i < action_names.size(); i++) {
      if (action_names[i] == "noop") {
        noop_idx = i;
        break;
      }
    }

    if (noop_idx == -1) {
      std::cout << "No Noop action found, using action 2" << std::endl;  // Updated to use index 2 (Noop)
      noop_idx = 2;  // Use Noop action which is at index 2 according to the output
    } else {
      std::cout << "Noop action found at index " << noop_idx << std::endl;
    }

    // Create actions array with minimal configuration
    std::cout << "Creating actions array..." << std::endl;
    int num_agents = 2;                                      // Assuming 2 agents from test_utils::create_test_grid
    ActionsType* actions = new ActionsType[num_agents * 2];  // Allocate flat array with 2 values per agent

    for (int i = 0; i < num_agents; i++) {
      int idx = i * 2;          // Calculate index in flat array
      actions[idx] = noop_idx;  // Use Noop action or action 2
      actions[idx + 1] = 0;     // No argument
      std::cout << " Agent " << i << " action: [" << actions[idx] << ", " << actions[idx + 1] << "]" << std::endl;
    }
    std::cout << "Actions array created." << std::endl;

    // Get agent information
    std::cout << "Agent information:" << std::endl;
    const auto& agents = grid->get_agents();
    for (size_t i = 0; i < agents.size(); i++) {
      std::cout << "  Agent " << i << " (id=" << agents[i]->id << "):" << std::endl;
      std::cout << "    Location: (" << agents[i]->location.r << ", " << agents[i]->location.c << ", "
                << agents[i]->location.layer << ")" << std::endl;
      std::cout << "    Group: " << agents[i]->group << std::endl;
    }

    // Perform the step operation
    std::cout << "ABOUT TO STEP THE GRID - critical operation" << std::endl;
    grid->step(actions);
    std::cout << "Grid step completed successfully!" << std::endl;

    // Check if rewards were updated
    std::cout << "Checking rewards after step:" << std::endl;
    for (size_t i = 0; i < reward_size; i++) {
      std::cout << "  Reward " << i << ": " << rewards[i] << std::endl;
    }

    // Access internal episode rewards for validation
    float* episode_rewards = grid->get_episode_rewards();
    if (episode_rewards != nullptr) {
      std::cout << "Episode rewards after step:" << std::endl;
      for (size_t i = 0; i < grid->num_agents(); i++) {
        std::cout << "  Episode Reward " << i << ": " << episode_rewards[i] << std::endl;
      }
    }

    // Access internal group rewards for validation
    float* group_rewards = grid->get_group_rewards();
    if (group_rewards != nullptr) {
      std::cout << "Group rewards after step:" << std::endl;
      // We don't know the exact size of group_rewards, so we'll just print the first few
      for (size_t i = 0; i < 3 && i < agents.size(); i++) {
        std::cout << "  Group Reward " << i << ": " << group_rewards[i] << std::endl;
      }
    }

    // Clean up
    std::cout << "Cleaning up..." << std::endl;
    delete[] actions;

    delete[] observations;
    delete[] terminals;
    delete[] truncations;
    delete[] rewards;
    // No need to delete episode_rewards or group_rewards as they are managed internally

    std::cout << "Test completed successfully!" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    FAIL() << "Exception occurred during test";
  } catch (...) {
    std::cerr << "UNKNOWN EXCEPTION occurred" << std::endl;
    FAIL() << "Unknown exception occurred during test";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}