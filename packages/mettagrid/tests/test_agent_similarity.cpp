// test_agent_similarity.cpp - Unit tests for compute_agent_similarity function
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "action_distance.hpp"
#include "action_handler.hpp"
#include "actions/move.hpp"
#include "actions/noop.hpp"
#include "actions/rotate.hpp"
#include "grid.hpp"
#include "matrix_profile.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class AgentSimilarityTest : public ::testing::Test {
protected:
  std::unique_ptr<Grid> grid;
  std::vector<std::unique_ptr<ActionHandler>> action_handlers;
  std::unique_ptr<ActionDistance::ActionDistanceLUT> distance_lut;
  std::unique_ptr<MatrixProfile::MatrixProfiler> profiler;

  static constexpr int GRID_SIZE = 20;

  void SetUp() override {
    grid = std::make_unique<Grid>(GRID_SIZE, GRID_SIZE);

    // Setup action handlers - using only basic actions for testing
    ActionConfig basic_config({}, {});

    action_handlers.push_back(std::make_unique<Noop>(basic_config));
    action_handlers.push_back(std::make_unique<Move>(basic_config));
    action_handlers.push_back(std::make_unique<Rotate>(basic_config));

    for (auto& handler : action_handlers) {
      handler->init(grid.get());
    }

    // Initialize distance LUT
    distance_lut = std::make_unique<ActionDistance::ActionDistanceLUT>();
    distance_lut->register_actions(action_handlers);

    // Initialize matrix profiler
    MatrixProfile::MatrixProfileConfig config;
    config.force_cpu = true;  // Force CPU for deterministic tests
    config.window_sizes = {5, 10, 20};
    profiler = std::make_unique<MatrixProfile::MatrixProfiler>(config);
    profiler->initialize(*distance_lut);
  }

  Agent* CreateAgent(int row, int col, int group_id = 1) {
    std::map<uint8_t, InventoryQuantity> resource_limits;
    std::map<uint8_t, RewardType> rewards;
    std::map<uint8_t, RewardType> resource_reward_max;

    AgentConfig config(0,
                       "test_agent",
                       static_cast<uint8_t>(group_id),
                       "test_group",
                       0,
                       0.0f,
                       resource_limits,
                       rewards,
                       resource_reward_max,
                       {},
                       {},
                       0.0f);

    Agent* agent = new Agent(static_cast<GridCoord>(row), static_cast<GridCoord>(col), config);
    float reward = 0.0f;
    agent->init(&reward);
    grid->add_object(agent);
    return agent;
  }

  void RecordActionSequence(Agent* agent, const std::vector<std::pair<ActionType, ActionArg>>& sequence) {
    for (const auto& [action, arg] : sequence) {
      agent->record_action(action, arg);
    }
  }
};

TEST_F(AgentSimilarityTest, IdenticalAgents) {
  // Create two agents with identical action sequences
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  std::vector<std::pair<ActionType, ActionArg>> pattern = {
      {1, 0},  // Move forward
      {2, 2},  // Rotate left
      {1, 0},  // Move forward
      {2, 3},  // Rotate right
      {0, 0}   // Noop
  };

  // Record the same pattern multiple times
  for (int i = 0; i < 10; i++) {
    RecordActionSequence(agent1, pattern);
    RecordActionSequence(agent2, pattern);
  }

  // Compute profiles
  std::vector<Agent*> agents = {agent1, agent2};
  auto profiles = profiler->compute_profiles(agents, {5});

  ASSERT_EQ(profiles.size(), 2);

  // Compute similarity
  float similarity =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);

  // Should be very high (near 1.0) for identical sequences
  EXPECT_GT(similarity, 0.95f);
  std::cout << "Identical agents similarity: " << similarity << std::endl;
}

TEST_F(AgentSimilarityTest, CompletelyDifferentAgents) {
  // Create two agents with completely different patterns
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  // Agent 1: Movement focused
  std::vector<std::pair<ActionType, ActionArg>> pattern1 = {
      {1, 0}, {1, 0}, {1, 1}, {1, 0}, {1, 1}  // All moves
  };

  // Agent 2: Rotation focused
  std::vector<std::pair<ActionType, ActionArg>> pattern2 = {
      {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 0}  // All rotations
  };

  for (int i = 0; i < 10; i++) {
    RecordActionSequence(agent1, pattern1);
    RecordActionSequence(agent2, pattern2);
  }

  // Compute profiles
  std::vector<Agent*> agents = {agent1, agent2};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Compute similarity
  float similarity =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);

  // Should be low for completely different behaviors
  EXPECT_LT(similarity, 0.3f);
  std::cout << "Different agents similarity: " << similarity << std::endl;
}

TEST_F(AgentSimilarityTest, SimilarButNotIdenticalAgents) {
  // Create agents with similar but not identical patterns
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  // Both use move-rotate patterns but with slight variations
  std::vector<std::pair<ActionType, ActionArg>> pattern1 = {{1, 0}, {2, 2}, {1, 0}, {2, 3}, {0, 0}};

  std::vector<std::pair<ActionType, ActionArg>> pattern2 = {
      {1, 0}, {2, 3}, {1, 0}, {2, 2}, {0, 0}  // Same actions, different rotation order
  };

  for (int i = 0; i < 10; i++) {
    RecordActionSequence(agent1, pattern1);
    RecordActionSequence(agent2, pattern2);
  }

  // Compute profiles
  std::vector<Agent*> agents = {agent1, agent2};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Compute similarity
  float similarity =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);

  // Should be moderately high - same actions, just reordered
  EXPECT_GT(similarity, 0.5f);
  EXPECT_LT(similarity, 0.9f);
  std::cout << "Similar agents similarity: " << similarity << std::endl;
}

TEST_F(AgentSimilarityTest, SharedMotifsWithDifferentFrequencies) {
  // Agents share some patterns but use them with different frequencies
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  std::vector<std::pair<ActionType, ActionArg>> shared_motif = {{1, 0}, {1, 0}, {2, 2}, {1, 0}, {0, 0}};

  std::vector<std::pair<ActionType, ActionArg>> unique_motif1 = {{2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 0}};

  std::vector<std::pair<ActionType, ActionArg>> unique_motif2 = {{0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}};

  // Agent 1: Uses shared motif 70% of the time
  for (int i = 0; i < 7; i++) {
    RecordActionSequence(agent1, shared_motif);
  }
  for (int i = 0; i < 3; i++) {
    RecordActionSequence(agent1, unique_motif1);
  }

  // Agent 2: Uses shared motif 40% of the time
  for (int i = 0; i < 4; i++) {
    RecordActionSequence(agent2, shared_motif);
  }
  for (int i = 0; i < 6; i++) {
    RecordActionSequence(agent2, unique_motif2);
  }

  // Compute profiles
  std::vector<Agent*> agents = {agent1, agent2};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Compute similarity
  float similarity =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);

  // Should detect the shared motif despite different frequencies
  EXPECT_GT(similarity, 0.3f);
  EXPECT_LT(similarity, 0.7f);
  std::cout << "Shared motifs similarity: " << similarity << std::endl;
}

TEST_F(AgentSimilarityTest, DifferentWindowSizes) {
  // Test that similarity is computed correctly for different window sizes
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  // Create a longer repeating pattern
  std::vector<std::pair<ActionType, ActionArg>> long_pattern = {
      {1, 0}, {1, 0}, {2, 2}, {1, 0}, {2, 3}, {0, 0}, {1, 1}, {2, 0}, {1, 0}, {0, 0}};

  for (int i = 0; i < 5; i++) {
    RecordActionSequence(agent1, long_pattern);
    RecordActionSequence(agent2, long_pattern);
  }

  // Compute profiles with multiple window sizes
  std::vector<Agent*> agents = {agent1, agent2};
  std::vector<uint8_t> window_sizes = {5, 10, 20};
  auto profiles = profiler->compute_profiles(agents, window_sizes);

  // Test similarity at different scales
  for (uint8_t window_size : window_sizes) {
    float similarity = MatrixProfile::Analysis::compute_agent_similarity(
        profiles[0], profiles[1], window_size, agent1, agent2, *distance_lut);

    // Should be high for all window sizes with identical patterns
    EXPECT_GT(similarity, 0.9f) << "Window size: " << window_size;
    std::cout << "Window " << window_size << " similarity: " << similarity << std::endl;
  }
}

TEST_F(AgentSimilarityTest, EmptyOrShortHistories) {
  // Test edge cases with empty or very short histories
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);
  Agent* agent3 = CreateAgent(15, 15);

  // Agent 1: No history
  // Agent 2: Very short history
  agent2->record_action(1, 0);
  agent2->record_action(2, 2);

  // Agent 3: History shorter than window size
  agent3->record_action(1, 0);
  agent3->record_action(2, 2);
  agent3->record_action(1, 0);

  std::vector<Agent*> agents = {agent1, agent2, agent3};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Similarity between agents with insufficient history should be 0
  if (profiles.size() >= 2) {
    float similarity =
        MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);
    EXPECT_EQ(similarity, 0.0f);
  }
}

TEST_F(AgentSimilarityTest, RankImportance) {
  // Test that the similarity function properly weights motif importance by rank
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);
  Agent* agent3 = CreateAgent(15, 15);

  // Primary motif (most common)
  std::vector<std::pair<ActionType, ActionArg>> primary = {{1, 0}, {1, 0}, {2, 2}, {1, 0}, {0, 0}};

  // Secondary motif
  std::vector<std::pair<ActionType, ActionArg>> secondary = {{2, 0}, {2, 1}, {0, 0}, {2, 2}, {2, 3}};

  // Agent 1 & 2: Share the same primary motif
  for (int i = 0; i < 8; i++) {
    RecordActionSequence(agent1, primary);
    RecordActionSequence(agent2, primary);
  }
  for (int i = 0; i < 2; i++) {
    RecordActionSequence(agent1, secondary);
  }

  // Agent 3: Different primary motif
  for (int i = 0; i < 8; i++) {
    RecordActionSequence(agent3, secondary);
  }
  for (int i = 0; i < 2; i++) {
    RecordActionSequence(agent3, primary);
  }

  std::vector<Agent*> agents = {agent1, agent2, agent3};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Agents 1 & 2 should be more similar to each other than to agent 3
  float sim_12 =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);
  float sim_13 =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[2], 5, agent1, agent3, *distance_lut);
  float sim_23 =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[1], profiles[2], 5, agent2, agent3, *distance_lut);

  EXPECT_GT(sim_12, sim_13) << "Agents with same primary motif should be more similar";
  EXPECT_GT(sim_12, sim_23) << "Agents with same primary motif should be more similar";

  std::cout << "Same primary motif similarity: " << sim_12 << std::endl;
  std::cout << "Different primary motif similarities: " << sim_13 << ", " << sim_23 << std::endl;
}

TEST_F(AgentSimilarityTest, NoMotifs) {
  // Test case where agents have no motifs (e.g., very random behavior)
  Agent* agent1 = CreateAgent(5, 5);
  Agent* agent2 = CreateAgent(10, 10);

  // Fill with pseudo-random actions
  for (int i = 0; i < 50; i++) {
    agent1->record_action((i * 7) % 3, (i * 3) % 4);
    agent2->record_action((i * 11) % 3, (i * 5) % 4);
  }

  std::vector<Agent*> agents = {agent1, agent2};
  auto profiles = profiler->compute_profiles(agents, {5});

  // Even with no clear motifs, similarity should still be computable
  float similarity =
      MatrixProfile::Analysis::compute_agent_similarity(profiles[0], profiles[1], 5, agent1, agent2, *distance_lut);

  // Should be valid (between 0 and 1)
  EXPECT_GE(similarity, 0.0f);
  EXPECT_LE(similarity, 1.0f);
  std::cout << "Random behavior similarity: " << similarity << std::endl;
}

// Add this test to your CMakeLists.txt or test runner
// Example CMakeLists.txt entry:
// add_executable(test_agent_similarity test_agent_similarity.cpp)
// target_link_libraries(test_agent_similarity
//   mettagrid_core
//   gtest
//   gtest_main
// )
