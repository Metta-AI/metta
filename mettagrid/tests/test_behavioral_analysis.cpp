#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "action_distance.hpp"
#include "action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/move.hpp"
#include "actions/noop.hpp"
#include "actions/rotate.hpp"
#include "agent_behavior.hpp"
#include "grid.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

// Test fixture for behavioral analysis tests
class BehavioralAnalysisTest : public ::testing::Test {
protected:
  std::unique_ptr<Grid> grid;
  std::vector<std::unique_ptr<ActionHandler>> action_handlers;
  std::unique_ptr<ActionDistance::ActionDistanceLUT> distance_lut;
  std::unique_ptr<AgentBehavior::BehaviorAnalyzer> analyzer;

  // Test configuration
  static constexpr int GRID_SIZE = 20;
  static constexpr int NUM_TEST_AGENTS = 10;

  void SetUp() override {
    grid = std::make_unique<Grid>(GRID_SIZE, GRID_SIZE);

    // Set up action handlers (simplified version)
    SetupActionHandlers();

    // Initialize distance LUT
    distance_lut = std::make_unique<ActionDistance::ActionDistanceLUT>();
    distance_lut->register_actions(action_handlers);

    // Initialize behavior analyzer
    analyzer = std::make_unique<AgentBehavior::BehaviorAnalyzer>();
  }

  void TearDown() override {
    // Clean up in reverse order
    analyzer.reset();
    distance_lut.reset();
    action_handlers.clear();
    grid.reset();
  }

  void SetupActionHandlers() {
    // Create basic action handlers for testing
    action_handlers.push_back(std::make_unique<Noop>(ActionConfig({}, {})));
    action_handlers.push_back(std::make_unique<Move>(ActionConfig({}, {})));
    action_handlers.push_back(std::make_unique<Rotate>(ActionConfig({}, {})));

    // Initialize handlers with grid
    for (auto& handler : action_handlers) {
      handler->init(grid.get());
    }
  }

  AgentConfig CreateTestAgentConfig(int group_id = 1) {
    std::map<uint8_t, InventoryQuantity> resource_limits;
    std::map<uint8_t, RewardType> rewards;
    std::map<uint8_t, RewardType> resource_reward_max;

    return AgentConfig(0,                               // type_id
                       "test_agent",                    // type_name
                       static_cast<uint8_t>(group_id),  // group_id
                       "test_group",                    // group_name
                       0,                               // freeze_duration
                       0.0f,                            // action_failure_penalty
                       resource_limits,                 // resource_limits
                       rewards,                         // resource_rewards
                       resource_reward_max,             // resource_reward_max
                       {},                              // stat_rewards
                       {},                              // stat_reward_max
                       0.0f                             // group_reward_pct
    );
  }

  Agent* CreateAndAddAgent(int row, int col, int group_id = 1) {
    auto config = CreateTestAgentConfig(group_id);
    Agent* agent = new Agent(static_cast<GridCoord>(row), static_cast<GridCoord>(col), config);
    float reward = 0.0f;
    agent->init(&reward);
    grid->add_object(agent);
    return agent;
  }

  // Helper to create an agent with a specific action pattern
  void RecordActionPattern(Agent* agent,
                           const std::vector<std::pair<ActionType, ActionArg>>& pattern,
                           int repeats = 1) {
    for (int r = 0; r < repeats; r++) {
      for (const auto& [action, arg] : pattern) {
        agent->record_action(action, arg);
      }
    }
  }
};

// ==================== Action Distance Tests ====================

TEST_F(BehavioralAnalysisTest, ActionDistanceLUT_BasicFunctionality) {
  // Test encoding and decoding
  auto encoded_noop = distance_lut->encode_action(0, 0);       // NOOP
  auto encoded_move_fwd = distance_lut->encode_action(1, 0);   // Move forward
  auto encoded_move_back = distance_lut->encode_action(1, 1);  // Move back

  EXPECT_NE(encoded_noop, encoded_move_fwd);
  EXPECT_NE(encoded_move_fwd, encoded_move_back);

  // Test decoding
  auto [type1, arg1] = distance_lut->decode_action(encoded_move_fwd);
  EXPECT_EQ(type1, 1);  // Move action
  EXPECT_EQ(arg1, 0);   // Forward

  // Test distance calculation
  auto dist_same = distance_lut->get_encoded_distance(encoded_noop, encoded_noop);
  EXPECT_EQ(dist_same, 0);

  auto dist_move_types = distance_lut->get_encoded_distance(encoded_move_fwd, encoded_move_back);
  EXPECT_GT(dist_move_types, 0);  // Opposite moves should have distance

  auto dist_different_actions = distance_lut->get_encoded_distance(encoded_noop, encoded_move_fwd);
  EXPECT_GT(dist_different_actions, dist_move_types);  // Different action types should have larger distance
}

TEST_F(BehavioralAnalysisTest, ActionDistanceLUT_SequenceDistance) {
  // Create two sequences
  std::vector<ActionType> types1 = {0, 1, 1, 2};  // NOOP, Move, Move, Rotate
  std::vector<ActionArg> args1 = {0, 0, 1, 0};    // -, Forward, Back, Up

  std::vector<ActionType> types2 = {0, 1, 1, 2};  // Same actions
  std::vector<ActionArg> args2 = {0, 0, 0, 0};    // -, Forward, Forward, Up

  auto seq1 = distance_lut->encode_sequence(types1, args1);
  auto seq2 = distance_lut->encode_sequence(types2, args2);

  auto distance = distance_lut->sequence_distance(seq1, seq2);
  EXPECT_GT(distance, 0);  // Should have some distance due to different move args

  // Test identical sequences
  auto distance_same = distance_lut->sequence_distance(seq1, seq1);
  EXPECT_EQ(distance_same, 0);
}

TEST_F(BehavioralAnalysisTest, ActionDistanceLUT_HumanReadable) {
  // Test decoding to human-readable strings
  auto encoded_move_fwd = distance_lut->encode_action(1, 0);
  auto encoded_rotate_left = distance_lut->encode_action(2, 2);

  auto str_move = distance_lut->decode_to_string(encoded_move_fwd);
  EXPECT_NE(str_move.find("move"), std::string::npos);
  EXPECT_NE(str_move.find("fwd"), std::string::npos);

  auto str_rotate = distance_lut->decode_to_string(encoded_rotate_left);
  EXPECT_NE(str_rotate.find("rotate"), std::string::npos);
  EXPECT_NE(str_rotate.find("left"), std::string::npos);
}

// ==================== Agent History Tests ====================

TEST_F(BehavioralAnalysisTest, AgentHistory_Recording) {
  auto* agent = CreateAndAddAgent(5, 5);

  // Record some actions
  agent->record_action(0, 0);  // NOOP
  agent->record_action(1, 0);  // Move forward
  agent->record_action(2, 1);  // Rotate down

  EXPECT_EQ(agent->history_count, 3);

  // Test copying history
  std::vector<ActionType> actions(3);
  std::vector<ActionArg> args(3);
  agent->copy_history_to_buffers(actions.data(), args.data());

  EXPECT_EQ(actions[0], 0);
  EXPECT_EQ(actions[1], 1);
  EXPECT_EQ(actions[2], 2);
  EXPECT_EQ(args[0], 0);
  EXPECT_EQ(args[1], 0);
  EXPECT_EQ(args[2], 1);
}

TEST_F(BehavioralAnalysisTest, AgentHistory_RingBuffer) {
  auto* agent = CreateAndAddAgent(5, 5);

  // Fill beyond capacity to test ring buffer
  for (size_t i = 0; i < Agent::MAX_HISTORY_LENGTH + 100; i++) {
    agent->record_action(i % 3, i % 2);
  }

  EXPECT_EQ(agent->history_count, Agent::MAX_HISTORY_LENGTH);

  // Verify we can still copy the full history
  std::vector<ActionType> actions(Agent::MAX_HISTORY_LENGTH);
  std::vector<ActionArg> args(Agent::MAX_HISTORY_LENGTH);
  agent->copy_history_to_buffers(actions.data(), args.data());

  // The oldest entries should have been overwritten
  // We should have the most recent MAX_HISTORY_LENGTH entries
}

// ==================== Behavior Analyzer Tests ====================

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_Initialization) {
  // Test initialization with small number of agents
  EXPECT_NO_THROW(analyzer->initialize(action_handlers, 10));

  // Verify we can get behavior info
  auto info = get_behavior_analysis_info();
  EXPECT_FALSE(info.empty());

  // Check CUDA availability
  bool cuda_available = is_cuda_available();
  std::cout << "CUDA available: " << (cuda_available ? "Yes" : "No") << std::endl;
  std::cout << get_cuda_unavailable_message() << std::endl;
}

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_SimplePattern) {
  analyzer->initialize(action_handlers, 5);

  // Create agents with a simple repeating pattern
  std::vector<Agent*> agents;
  std::vector<std::pair<ActionType, ActionArg>> pattern = {
      {1, 0},  // Move forward
      {2, 2},  // Rotate left
      {1, 0},  // Move forward
      {2, 3},  // Rotate right
  };

  for (int i = 0; i < 5; i++) {
    auto* agent = CreateAndAddAgent(i, i);
    RecordActionPattern(agent, pattern, 10);  // Repeat pattern 10 times
    agents.push_back(agent);
  }

  // Analyze with small window sizes
  std::vector<int> window_sizes = {4, 8};
  auto motifs = analyzer->get_dominant_motifs(agents, window_sizes);

  // We should find our pattern as a dominant motif
  EXPECT_FALSE(motifs.empty());

  if (!motifs.empty()) {
    // The most dominant motif should have high prevalence
    EXPECT_GT(motifs[0].prevalence, 0.5f);  // At least 50% of agents
    EXPECT_EQ(motifs[0].window_size, 4);    // Should match our pattern length
  }
}

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_DiversePatterns) {
  analyzer->initialize(action_handlers, 10);

  std::vector<Agent*> agents;

  // Group 1: Move-heavy pattern (5 agents)
  std::vector<std::pair<ActionType, ActionArg>> move_pattern = {
      {1, 0}, {1, 0}, {1, 1}, {1, 0}  // Forward, forward, back, forward
  };

  for (int i = 0; i < 5; i++) {
    auto* agent = CreateAndAddAgent(i, 0, 1);
    RecordActionPattern(agent, move_pattern, 15);
    agents.push_back(agent);
  }

  // Group 2: Rotate-heavy pattern (5 agents)
  std::vector<std::pair<ActionType, ActionArg>> rotate_pattern = {
      {2, 0}, {2, 1}, {2, 2}, {2, 3}  // Rotate all directions
  };

  for (int i = 0; i < 5; i++) {
    auto* agent = CreateAndAddAgent(i, 10, 2);
    RecordActionPattern(agent, rotate_pattern, 15);
    agents.push_back(agent);
  }

  // Get behavior statistics
  auto stats = analyzer->get_behavior_stats(agents);

  // Check action frequencies
  EXPECT_GT(stats.action_frequencies["move"], 0.4f);    // ~50% move actions
  EXPECT_GT(stats.action_frequencies["rotate"], 0.4f);  // ~50% rotate actions

  // Behavioral diversity should be moderate (two distinct groups)
  EXPECT_GT(stats.behavioral_diversity, 0.5f);
  EXPECT_LT(stats.behavioral_diversity, 0.9f);
}

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_EmergentBehavior) {
  analyzer->initialize(action_handlers, 20);

  std::vector<Agent*> agents;

  // Simulate emergent behavior: agents start random then converge
  for (int i = 0; i < 20; i++) {
    auto* agent = CreateAndAddAgent(i % GRID_SIZE, (i * 2) % GRID_SIZE);

    // Phase 1: Random actions (exploration)
    for (int j = 0; j < 20; j++) {
      agent->record_action(j % 3, j % 2);
    }

    // Phase 2: Converged pattern (learned behavior)
    std::vector<std::pair<ActionType, ActionArg>> learned_pattern = {
        {1, 0}, {0, 0}, {1, 0}, {2, 2}  // Move, wait, move, turn left
    };
    RecordActionPattern(agent, learned_pattern, 20);

    agents.push_back(agent);
  }

  // Analyze with multiple window sizes
  std::vector<int> window_sizes = {4, 10, 20};
  auto motifs = analyzer->get_dominant_motifs(agents, window_sizes);

  // Should find patterns in the learned behavior
  bool found_learned_pattern = false;
  for (const auto& motif : motifs) {
    if (motif.window_size == 4 && motif.prevalence > 0.7f) {
      found_learned_pattern = true;
      break;
    }
  }

  EXPECT_TRUE(found_learned_pattern) << "Should detect the converged learned pattern";
}

// ==================== Performance Tests ====================

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_PerformanceTracking) {
  analyzer->initialize(action_handlers, 10);

  // Create agents with sufficient history
  std::vector<Agent*> agents;
  for (int i = 0; i < 10; i++) {
    auto* agent = CreateAndAddAgent(i, i);
    // Fill with enough data for analysis
    for (int j = 0; j < 100; j++) {
      agent->record_action(j % 3, j % 4);
    }
    agents.push_back(agent);
  }

  // Run analysis
  auto motifs = analyzer->get_dominant_motifs(agents, {10, 25});

  // Get performance stats
  auto perf = analyzer->get_performance_stats();

  EXPECT_GT(perf.total_time_ms, 0);
  EXPECT_GT(perf.total_comparisons, 0);

  std::cout << "Performance stats:\n"
            << "  Total time: " << perf.total_time_ms << " ms\n"
            << "  Comparisons: " << perf.total_comparisons << "\n"
            << "  Comparisons/sec: " << perf.comparisons_per_second << "\n";
}

// ==================== Edge Case Tests ====================

TEST_F(BehavioralAnalysisTest, BehaviorAnalyzer_EdgeCases) {
  analyzer->initialize(action_handlers, 10);

  // Test 1: Empty agent list
  std::vector<Agent*> empty_agents;
  auto motifs = analyzer->get_dominant_motifs(empty_agents, {10});
  EXPECT_TRUE(motifs.empty());

  // Test 2: Agents with no history
  std::vector<Agent*> no_history_agents;
  for (int i = 0; i < 3; i++) {
    auto* agent = CreateAndAddAgent(i, i);
    no_history_agents.push_back(agent);
  }
  motifs = analyzer->get_dominant_motifs(no_history_agents, {10});
  EXPECT_TRUE(motifs.empty());

  // Test 3: Agents with history shorter than window
  std::vector<Agent*> short_history_agents;
  for (int i = 0; i < 3; i++) {
    auto* agent = CreateAndAddAgent(i + 10, i);
    agent->record_action(0, 0);
    agent->record_action(1, 0);
    short_history_agents.push_back(agent);
  }
  motifs = analyzer->get_dominant_motifs(short_history_agents, {10});
  EXPECT_TRUE(motifs.empty());

  // Test 4: Single agent (no cross-agent patterns possible)
  std::vector<Agent*> single_agent;
  auto* agent = CreateAndAddAgent(15, 15);
  for (int i = 0; i < 50; i++) {
    agent->record_action(i % 2, 0);
  }
  single_agent.push_back(agent);

  motifs = analyzer->get_dominant_motifs(single_agent, {5});
  // Should still work but won't find dominant patterns across agents
  EXPECT_TRUE(motifs.empty() || motifs[0].prevalence <= 1.0f);
}

// ==================== Integration Tests ====================

TEST_F(BehavioralAnalysisTest, Integration_FullScenario) {
  // Initialize with realistic number of agents
  analyzer->initialize(action_handlers, 50);

  std::vector<Agent*> agents;

  // Simulate different agent strategies

  // Strategy 1: Explorers (constantly moving)
  for (int i = 0; i < 15; i++) {
    auto* agent = CreateAndAddAgent(i, i % 10, 1);
    for (int step = 0; step < 200; step++) {
      if (step % 4 == 0) {
        agent->record_action(2, step % 4);  // Rotate
      } else {
        agent->record_action(1, 0);  // Move forward
      }
    }
    agents.push_back(agent);
  }

  // Strategy 2: Defenders (mostly stationary with occasional rotation)
  for (int i = 0; i < 15; i++) {
    auto* agent = CreateAndAddAgent(i + 5, (i + 10) % GRID_SIZE, 2);
    for (int step = 0; step < 200; step++) {
      if (step % 10 == 0) {
        agent->record_action(2, 1);  // Rotate down
      } else {
        agent->record_action(0, 0);  // NOOP
      }
    }
    agents.push_back(agent);
  }

  // Strategy 3: Patrollers (fixed route pattern)
  std::vector<std::pair<ActionType, ActionArg>> patrol_pattern = {{1, 0},
                                                                  {1, 0},
                                                                  {2, 3},
                                                                  {1, 0},
                                                                  {1, 0},
                                                                  {2, 3},  // Square pattern
                                                                  {1, 0},
                                                                  {1, 0},
                                                                  {2, 3},
                                                                  {1, 0},
                                                                  {1, 0},
                                                                  {2, 3}};

  for (int i = 0; i < 20; i++) {
    auto* agent = CreateAndAddAgent((i * 2) % GRID_SIZE, (i * 3) % GRID_SIZE, 3);
    RecordActionPattern(agent, patrol_pattern, 16);
    agents.push_back(agent);
  }

  // Analyze behaviors
  std::vector<int> window_sizes = {5, 12, 25, 50};
  auto motifs = analyzer->get_dominant_motifs(agents, window_sizes);

  // Verify we found distinct patterns
  EXPECT_GE(motifs.size(), 3);  // Should find at least 3 different patterns

  // Get behavior statistics
  auto stats = analyzer->get_behavior_stats(agents);

  // Verify action distribution makes sense
  EXPECT_GT(stats.action_frequencies["noop"], 0.1f);    // Defenders
  EXPECT_GT(stats.action_frequencies["move"], 0.3f);    // Explorers and patrollers
  EXPECT_GT(stats.action_frequencies["rotate"], 0.1f);  // All groups

  // Print summary for manual inspection
  std::cout << "\nIntegration Test Results:\n";
  std::cout << "Found " << motifs.size() << " dominant motifs\n";

  for (size_t i = 0; i < std::min(size_t(5), motifs.size()); i++) {
    const auto& motif = motifs[i];
    std::cout << "  Motif " << i + 1 << ": "
              << "window=" << motif.window_size << ", prevalence=" << motif.prevalence
              << ", agents=" << motif.agent_ids.size() << ", pattern=" << motif.pattern_description << "\n";
  }

  std::cout << "\nAction frequencies:\n";
  for (const auto& [action, freq] : stats.action_frequencies) {
    std::cout << "  " << action << ": " << freq << "\n";
  }
  std::cout << "Behavioral diversity: " << stats.behavioral_diversity << "\n";
}
