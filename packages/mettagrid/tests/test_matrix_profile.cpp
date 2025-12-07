#include <gtest/gtest.h>

#include <chrono>
#include <numeric>
#include <random>

#include "action_distance.hpp"
#include "action_handler.hpp"
#include "matrix_profile.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// Mock action handlers for testing
class MockActionHandler : public ActionHandler {
public:
  MockActionHandler(const std::string& name, uint8_t max_arg_val)
      : ActionHandler(ActionConfig({}, {}), name), max_arg_(max_arg_val) {}

  uint8_t max_arg() const override {
    return max_arg_;
  }

protected:
  bool _handle_action(Agent*, ActionArg) override {
    return true;
  }

private:
  uint8_t max_arg_;
};

class MatrixProfileTest : public ::testing::Test {
protected:
  std::unique_ptr<ActionDistance::ActionDistanceLUT> distance_lut;
  std::unique_ptr<MatrixProfile::MatrixProfiler> profiler;

  void SetUp() override {
    // Create mock action handlers
    std::vector<std::unique_ptr<ActionHandler>> handlers;
    handlers.push_back(std::make_unique<MockActionHandler>("noop", 0));
    handlers.push_back(std::make_unique<MockActionHandler>("move", 1));
    handlers.push_back(std::make_unique<MockActionHandler>("rotate", 3));
    handlers.push_back(std::make_unique<MockActionHandler>("attack", 8));

    // Initialize distance LUT
    distance_lut = std::make_unique<ActionDistance::ActionDistanceLUT>();
    distance_lut->register_actions(handlers);

    // Create matrix profiler with test config
    MatrixProfile::MatrixProfileConfig config;
    config.window_sizes = {4, 8, 16};
    config.min_window_size = 3;
    config.max_window_size = 32;

    profiler = std::make_unique<MatrixProfile::MatrixProfiler>(config);
  }

  // Helper to create a mock agent with action history
  std::unique_ptr<Agent> CreateMockAgent(GridObjectId id,
                                         const std::vector<std::pair<ActionType, ActionArg>>& history) {
    // Minimal agent config for testing
    std::map<uint8_t, InventoryQuantity> limits;
    std::map<uint8_t, RewardType> rewards;
    AgentConfig config(0, "test", 1, "test", 0, 0.0f, limits, rewards, {}, {}, {}, 0.0f);

    auto agent = std::make_unique<Agent>(0, 0, config);
    agent->agent_id = id;

    float reward = 0.0f;
    agent->init(&reward);

    // Record the history
    for (const auto& [action, arg] : history) {
      agent->record_action(action, arg);
    }

    return agent;
  }

  // Generate a repeating pattern
  std::vector<std::pair<ActionType, ActionArg>> GenerateRepeatingPattern(
      const std::vector<std::pair<ActionType, ActionArg>>& pattern,
      int repeats) {
    std::vector<std::pair<ActionType, ActionArg>> result;
    result.reserve(pattern.size() * static_cast<size_t>(repeats));

    for (int i = 0; i < repeats; i++) {
      result.insert(result.end(), pattern.begin(), pattern.end());
    }

    return result;
  }

  // Generate random actions
  std::vector<std::pair<ActionType, ActionArg>> GenerateRandomActions(int count, int seed = 42) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<int> action_dist(0, 3);
    std::uniform_int_distribution<int> arg_dist(0, 3);

    std::vector<std::pair<ActionType, ActionArg>> result;
    result.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; i++) {
      result.push_back({action_dist(rng), arg_dist(rng)});
    }

    return result;
  }
};

// ==================== Basic Functionality Tests ====================

TEST_F(MatrixProfileTest, Initialization) {
  EXPECT_NO_THROW(profiler->initialize(*distance_lut));

  // Should start with no GPU memory usage in CPU mode
  EXPECT_EQ(profiler->get_gpu_memory_usage(), 0);
}

TEST_F(MatrixProfileTest, EmptyAgentList) {
  profiler->initialize(*distance_lut);

  std::vector<Agent*> empty_agents;
  auto profiles = profiler->compute_profiles(empty_agents);

  EXPECT_TRUE(profiles.empty());
}

TEST_F(MatrixProfileTest, SingleAgentSimplePattern) {
  profiler->initialize(*distance_lut);

  // Create agent with simple repeating pattern
  std::vector<std::pair<ActionType, ActionArg>> pattern = {
      {1, 0}, {1, 1}, {2, 0}, {0, 0}  // move fwd, move back, rotate up, noop
  };
  auto history = GenerateRepeatingPattern(pattern, 10);

  auto agent = CreateMockAgent(1, history);
  std::vector<Agent*> agents = {agent.get()};

  // Compute profiles with window size 4 (matching pattern length)
  auto profiles = profiler->compute_profiles(agents, {4});

  ASSERT_EQ(profiles.size(), 1);
  ASSERT_EQ(profiles[0].agent_id, 1);
  ASSERT_FALSE(profiles[0].window_results.empty());

  // Check that we found repeating patterns (low distances)
  const auto& window_result = profiles[0].window_results[0];
  EXPECT_EQ(window_result.window_size, 4);

  // Most distances should be 0 (exact matches)
  int zero_distances = 0;
  for (uint16_t dist : window_result.distances) {
    if (dist == 0) zero_distances++;
  }

  // We should have many exact matches due to repetition
  EXPECT_GT(zero_distances, history.size() / 2);
}

TEST_F(MatrixProfileTest, MultipleAgentsIdenticalPatterns) {
  profiler->initialize(*distance_lut);

  // Create multiple agents with the same pattern
  std::vector<std::pair<ActionType, ActionArg>> pattern = {
      {1, 0}, {2, 2}, {1, 0}, {2, 3}  // move-rotate-move-rotate
  };

  std::vector<std::unique_ptr<Agent>> agent_objects;
  std::vector<Agent*> agents;

  for (int i = 0; i < 5; i++) {
    auto history = GenerateRepeatingPattern(pattern, 15);
    auto agent = CreateMockAgent(static_cast<GridObjectId>(i + 1), history);
    agents.push_back(agent.get());
    agent_objects.push_back(std::move(agent));
  }

  // Compute profiles
  auto profiles = profiler->compute_profiles(agents, {4});

  ASSERT_EQ(profiles.size(), 5);

  // All agents should have similar profile characteristics
  for (const auto& profile : profiles) {
    ASSERT_FALSE(profile.window_results.empty());
    const auto& motifs = profile.window_results[0].top_motifs;
    ASSERT_FALSE(motifs.empty());

    // Top motif should have very low distance (pattern repeats)
    EXPECT_LT(motifs[0].distance, 5);
  }
}

TEST_F(MatrixProfileTest, RandomVsPatternedBehavior) {
  profiler->initialize(*distance_lut);

  std::vector<std::unique_ptr<Agent>> agent_objects;
  std::vector<Agent*> agents;

  // Create agents with random behavior
  for (int i = 0; i < 3; i++) {
    auto history = GenerateRandomActions(100, i);
    auto agent = CreateMockAgent(static_cast<GridObjectId>(i + 1), history);
    agents.push_back(agent.get());
    agent_objects.push_back(std::move(agent));
  }

  // Create agents with patterned behavior
  std::vector<std::pair<ActionType, ActionArg>> pattern = {{1, 0}, {1, 0}, {2, 1}, {0, 0}, {2, 3}, {1, 1}};

  for (int i = 0; i < 3; i++) {
    auto history = GenerateRepeatingPattern(pattern, 16);
    auto agent = CreateMockAgent(static_cast<GridObjectId>(i + 4), history);
    agents.push_back(agent.get());
    agent_objects.push_back(std::move(agent));
  }

  // Compute profiles
  auto profiles = profiler->compute_profiles(agents, {6});

  ASSERT_EQ(profiles.size(), 6);

  // Random agents should have higher average distances
  float avg_random_dist = 0;
  for (int i = 0; i < 3; i++) {
    const auto& dists = profiles[static_cast<size_t>(i)].window_results[0].distances;
    float sum = std::accumulate(dists.begin(), dists.end(), 0.0f);
    avg_random_dist += sum / dists.size();
  }
  avg_random_dist /= 3;

  // Patterned agents should have lower average distances
  float avg_pattern_dist = 0;
  for (int i = 3; i < 6; i++) {
    const auto& dists = profiles[static_cast<size_t>(i)].window_results[0].distances;
    float sum = std::accumulate(dists.begin(), dists.end(), 0.0f);
    avg_pattern_dist += sum / dists.size();
  }
  avg_pattern_dist /= 3;

  // Patterned behavior should have significantly lower distances
  EXPECT_LT(avg_pattern_dist, avg_random_dist * 0.5f);
}

// ==================== Cross-Agent Pattern Tests ====================

TEST_F(MatrixProfileTest, CrossAgentPatterns_SharedMotifs) {
  profiler->initialize(*distance_lut);

  // Create agents that share a common subsequence
  std::vector<std::pair<ActionType, ActionArg>> shared_motif = {{1, 0}, {1, 0}, {2, 2}, {0, 0}};

  std::vector<std::unique_ptr<Agent>> agent_objects;
  std::vector<Agent*> agents;

  // Agent 1: shared motif embedded in random actions
  auto history1 = GenerateRandomActions(20, 1);
  history1.insert(history1.begin() + 10, shared_motif.begin(), shared_motif.end());
  auto agent1 = CreateMockAgent(1, history1);
  agents.push_back(agent1.get());
  agent_objects.push_back(std::move(agent1));

  // Agent 2: shared motif at different position
  auto history2 = GenerateRandomActions(20, 2);
  history2.insert(history2.begin() + 5, shared_motif.begin(), shared_motif.end());
  auto agent2 = CreateMockAgent(2, history2);
  agents.push_back(agent2.get());
  agent_objects.push_back(std::move(agent2));

  // Find cross-agent patterns
  auto patterns = profiler->find_cross_agent_patterns(agents, 4, 5.0f);

  // Should find the shared motif
  EXPECT_FALSE(patterns.shared_motifs.empty());

  bool found_exact_match = false;
  for (const auto& motif : patterns.shared_motifs) {
    if (motif.distance == 0 && motif.length == 4) {
      found_exact_match = true;
      break;
    }
  }

  EXPECT_TRUE(found_exact_match) << "Should find the exact shared subsequence";
}

// ==================== Performance Tests ====================

TEST_F(MatrixProfileTest, PerformanceScaling) {
  profiler->initialize(*distance_lut);

  // Test with increasing numbers of agents
  std::vector<int> agent_counts = {10, 25, 50};
  std::vector<float> compute_times;

  for (int count : agent_counts) {
    std::vector<std::unique_ptr<Agent>> agent_objects;
    std::vector<Agent*> agents;

    // Create agents with moderate-length histories
    for (int i = 0; i < count; i++) {
      auto history = GenerateRandomActions(100, i);
      auto agent = CreateMockAgent(static_cast<GridObjectId>(i + 1), history);
      agents.push_back(agent.get());
      agent_objects.push_back(std::move(agent));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto profiles = profiler->compute_profiles(agents, {10});
    auto end = std::chrono::high_resolution_clock::now();

    float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    compute_times.push_back(time_ms);

    auto stats = profiler->get_last_performance_stats();

    std::cout << "Agents: " << count << ", Time: " << time_ms << " ms"
              << ", Comparisons: " << stats.total_comparisons << ", Rate: " << stats.comparisons_per_second
              << " /sec\n";
  }

  // Verify reasonable scaling (shouldn't be worse than O(n²))
  if (compute_times.size() >= 2) {
    float ratio = compute_times.back() / compute_times[0];
    float agent_ratio = static_cast<float>(agent_counts.back()) / agent_counts[0];

    // Time should scale at most quadratically with agent count
    EXPECT_LT(ratio, agent_ratio * agent_ratio * 2.0f);
  }
}

// ==================== Window Size Tests ====================

TEST_F(MatrixProfileTest, MultipleWindowSizes) {
  profiler->initialize(*distance_lut);

  // Create agent with multi-scale patterns
  std::vector<std::pair<ActionType, ActionArg>> small_pattern = {{1, 0}, {2, 1}};
  std::vector<std::pair<ActionType, ActionArg>> large_pattern = {
      {1, 0}, {1, 0}, {2, 1}, {0, 0}, {2, 3}, {1, 1}, {0, 0}, {0, 0}};

  std::vector<std::pair<ActionType, ActionArg>> history;
  // Alternate between patterns
  for (int i = 0; i < 10; i++) {
    history.insert(history.end(), small_pattern.begin(), small_pattern.end());
    if (i % 2 == 0) {
      history.insert(history.end(), large_pattern.begin(), large_pattern.end());
    }
  }

  auto agent = CreateMockAgent(1, history);
  std::vector<Agent*> agents = {agent.get()};

  // Test multiple window sizes
  std::vector<uint8_t> windows = {2, 4, 8};
  auto profiles = profiler->compute_profiles(agents, windows);

  ASSERT_EQ(profiles.size(), 1);
  ASSERT_EQ(profiles[0].window_results.size(), 3);

  // Each window size should find different patterns
  for (const auto& window_result : profiles[0].window_results) {
    EXPECT_FALSE(window_result.top_motifs.empty());

    // Smaller windows should find more matches
    if (window_result.window_size == 2) {
      int low_dist_count = 0;
      for (uint16_t dist : window_result.distances) {
        if (dist < 5) low_dist_count++;
      }
      EXPECT_GT(low_dist_count, history.size() / 4);
    }
  }
}

// ==================== Edge Cases ====================

TEST_F(MatrixProfileTest, EdgeCases) {
  profiler->initialize(*distance_lut);

  // Test 1: History shorter than window size
  auto short_agent = CreateMockAgent(1, {{1, 0}, {2, 1}});
  std::vector<Agent*> agents = {short_agent.get()};

  auto profiles = profiler->compute_profiles(agents, {10});
  EXPECT_TRUE(profiles.empty() || profiles[0].window_results.empty());

  // Test 2: Window size of 1 (should work but not very meaningful)
  auto normal_agent = CreateMockAgent(2, GenerateRandomActions(50));
  agents = {normal_agent.get()};

  profiles = profiler->compute_profiles(agents, {1});
  ASSERT_FALSE(profiles.empty());
  ASSERT_FALSE(profiles[0].window_results.empty());

  // Test 3: Very large window size
  profiles = profiler->compute_profiles(agents, {100});
  EXPECT_TRUE(profiles.empty() || profiles[0].window_results.empty());
}

// ==================== Analysis Function Tests ====================

TEST_F(MatrixProfileTest, TopMotifExtraction) {
  // Test the motif extraction function directly
  std::vector<uint16_t> distances = {10, 5, 15, 3, 20, 8, 2, 12, 6, 18};
  std::vector<uint32_t> indices = {5, 8, 2, 9, 1, 0, 4, 3, 7, 6};

  auto motifs = MatrixProfile::Analysis::find_top_motifs(distances, indices, 3, 3);

  ASSERT_LE(motifs.size(), 3);  // Should return at most 3 motifs

  if (!motifs.empty()) {
    // First motif should be the minimum distance
    EXPECT_EQ(motifs[0].distance, 2);
    EXPECT_EQ(motifs[0].start_idx, 6);
    EXPECT_EQ(motifs[0].match_idx, 4);
  }
}

// ==================== Incremental Update Tests ====================

TEST_F(MatrixProfileTest, IncrementalUpdates) {
  profiler->initialize(*distance_lut);

  // Create initial agent
  auto agent = CreateMockAgent(1, GenerateRandomActions(50));

  // Test single agent update (should be no-op in CPU mode)
  EXPECT_NO_THROW(profiler->update_agent(agent.get()));

  // Test batch update
  std::vector<Agent*> agents = {agent.get()};
  EXPECT_NO_THROW(profiler->batch_update(agents));

  // Clear cache
  EXPECT_NO_THROW(profiler->clear_cache());
}
