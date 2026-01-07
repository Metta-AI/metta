#include <benchmark/benchmark.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "actions/attack.hpp"
#include "actions/change_vibe.hpp"
#include "bindings/mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/agent_config.hpp"
#include "objects/collective_config.hpp"
#include "objects/wall.hpp"

namespace py = pybind11;

// Global pointer to store the Python interpreter guard
// This ensures it stays alive for the entire program duration
static std::unique_ptr<py::scoped_interpreter> g_python_guard;

// TODO: Currently this benchmark requires Python/pybind11 because the MettaGrid
// API is tightly coupled with Python types (py::dict, py::list, py::array_t).
//
// The goal is to refactor MettaGrid to have a pure C++ core with a thin Python
// wrapper layer. When that refactoring is complete, we should be able to:
// 1. Remove all pybind11 includes from this benchmark
// 2. Remove the Python interpreter initialization
// 3. Work directly with C++ types (std::vector, std::unordered_map, etc.)
//
// We'll know we've succeeded when this file has zero references to pybind11!

// Helper functions for creating configuration and map
GameConfig CreateBenchmarkConfig(size_t num_agents) {
  std::vector<std::string> resource_names = {"ore", "heart"};

  std::shared_ptr<ActionConfig> action_cfg = std::make_shared<ActionConfig>(
      std::unordered_map<InventoryItem, InventoryQuantity>(), std::unordered_map<InventoryItem, InventoryQuantity>());

  std::shared_ptr<AttackActionConfig> attack_cfg =
      std::make_shared<AttackActionConfig>(std::unordered_map<InventoryItem, InventoryQuantity>(),
                                           std::unordered_map<InventoryItem, InventoryQuantity>(),
                                           std::unordered_map<InventoryItem, InventoryQuantity>(),
                                           std::unordered_map<InventoryItem, InventoryQuantity>(),
                                           std::unordered_map<InventoryItem, InventoryQuantity>());

  std::shared_ptr<ChangeVibeActionConfig> change_vibe_cfg =
      std::make_shared<ChangeVibeActionConfig>(std::unordered_map<InventoryItem, InventoryQuantity>(),
                                               std::unordered_map<InventoryItem, InventoryQuantity>(),
                                               4);

  // GameConfig expects an unordered_map for actions
  std::unordered_map<std::string, std::shared_ptr<ActionConfig>> actions_cfg;

  actions_cfg["noop"] = action_cfg;
  actions_cfg["move"] = action_cfg;
  actions_cfg["rotate"] = action_cfg;
  actions_cfg["attack"] = attack_cfg;
  actions_cfg["change_vibe"] = change_vibe_cfg;

  std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>> objects_cfg;

  objects_cfg["wall"] = std::make_shared<WallConfig>(1, "wall", false);
  objects_cfg["agent.team1"] = std::make_shared<AgentConfig>(0, "agent", 0, "team1");
  objects_cfg["agent.team2"] = std::make_shared<AgentConfig>(0, "agent", 1, "team2");

  // Create default global observation config
  GlobalObsConfig global_obs_config;
  global_obs_config.episode_completion_pct = true;
  global_obs_config.last_action = true;
  global_obs_config.last_reward = true;

  // Empty vibe_names and feature_ids for benchmark
  std::vector<std::string> vibe_names;
  std::unordered_map<std::string, ObservationType> feature_ids;
  std::unordered_map<int, std::string> tag_id_map;

  return GameConfig(num_agents,
                    10000,
                    false,
                    11,
                    11,
                    resource_names,
                    vibe_names,
                    100,
                    global_obs_config,
                    feature_ids,
                    actions_cfg,
                    objects_cfg,
                    tag_id_map,
                    std::unordered_map<std::string, std::shared_ptr<CollectiveConfig>>(),
                    false,
                    std::unordered_map<std::string, float>(),
                    0,
                    nullptr);
}

py::list CreateDefaultMap(size_t num_agents_per_team = 2) {
  py::list map;
  const int width = 32;
  const int height = 32;

  // First, create empty map
  for (int r = 0; r < height; ++r) {
    py::list row;
    for (int c = 0; c < width; ++c) {
      if (r == 0 || r == height - 1 || c == 0 || c == width - 1) {
        row.append("wall");
      } else {
        row.append(".");
      }
    }
    map.append(row);
  }

  // Place agents symmetrically
  size_t agents_placed = 0;
  std::vector<std::pair<size_t, size_t>> positions = {
      {8, 8},   {8, 24},  {24, 8},  {24, 24}, {16, 8},  {16, 24}, {8, 16},  {24, 16}, {12, 12},
      {12, 20}, {20, 12}, {20, 20}, {16, 16}, {16, 12}, {16, 20}, {12, 16}, {20, 16}, {10, 10},
      {10, 22}, {22, 10}, {22, 22}, {14, 14}, {14, 18}, {18, 14}, {18, 18}};

  for (size_t i = 0; i < positions.size() && agents_placed < num_agents_per_team * 2; ++i) {
    auto [r, c] = positions[i];
    std::string team = (agents_placed % 2 == 0) ? "agent.team1" : "agent.team2";
    py::list row = map[r].cast<py::list>();
    row[c] = team;
    agents_placed++;
  }

  return map;
}

// Utility function to generate valid random actions
// Based on CreateBenchmarkConfig: noop=1, move=4, rotate=4, attack=1, swap=1, change_vibe=1
// Note that rotate and swap no longer exist, so the actions taken below might be meaningless.
// Total: 12 actions
py::array_t<int> GenerateValidRandomActions(size_t num_agents, std::mt19937* gen) {
  const size_t num_flat_actions = 12;  // Hardcoded based on benchmark config

  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(num_agents)};
  py::array_t<int> actions(shape);
  auto* actions_ptr = static_cast<int*>(actions.request().ptr);

  std::uniform_int_distribution<> action_dist(0, static_cast<int>(num_flat_actions) - 1);
  for (size_t i = 0; i < num_agents; ++i) {
    actions_ptr[i] = action_dist(*gen);
  }

  return actions;
}

// Pre-generate a sequence of actions for the benchmark
std::vector<py::array_t<int>> PreGenerateActionSequence(size_t num_agents, size_t sequence_length) {
  std::vector<py::array_t<int>> action_sequence;
  action_sequence.reserve(static_cast<size_t>(sequence_length));

  // Use a deterministic seed for reproducible benchmarks
  std::mt19937 gen(42);

  for (size_t i = 0; i < sequence_length; ++i) {
    action_sequence.push_back(GenerateValidRandomActions(num_agents, &gen));
  }

  return action_sequence;
}

// Benchmark fixture class to ensure proper setup/teardown
class MettaGridBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State&) override {
    // Ensure Python is initialized
    if (!g_python_guard) {
      setup_error = "Python interpreter not initialized";
      return;
    }

    // Acquire GIL for setup
    py::gil_scoped_acquire acquire;

    // Setup with default 4 agents (matching Python benchmark config)
    num_agents = 4;
    auto cfg = CreateBenchmarkConfig(num_agents);
    auto map = CreateDefaultMap(2);

    env = std::make_unique<MettaGrid>(cfg, map, 42);

    // Initialize buffers for the environment
    // Observations: [num_agents, num_tokens, 3]
    const size_t num_tokens = cfg.num_observation_tokens;
    std::vector<py::ssize_t> obs_shape = {
        static_cast<py::ssize_t>(num_agents), static_cast<py::ssize_t>(num_tokens), 3};
    auto observations = py::array_t<uint8_t, py::array::c_style>(obs_shape);
    auto terminals = py::array_t<bool, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    auto truncations = py::array_t<bool, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    auto rewards = py::array_t<float, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    actions_buffer = py::array_t<int, py::array::c_style>(static_cast<py::ssize_t>(num_agents));

    // Initialize actions to zero
    std::fill(static_cast<int*>(actions_buffer.request().ptr),
              static_cast<int*>(actions_buffer.request().ptr) + actions_buffer.size(),
              0);

    env->set_buffers(observations, terminals, truncations, rewards, actions_buffer);

    // Pre-generate action sequence matching Python benchmark
    // Python uses: iterations = 1000, rounds = 20, total = 20000
    const int iterations = 1000;
    const int rounds = 20;
    const int total_iterations = iterations * rounds;

    action_sequence = PreGenerateActionSequence(num_agents, total_iterations);
    iteration_counter = 0;
    setup_error.clear();
  }

  void TearDown(const ::benchmark::State&) override {
    // Acquire GIL for cleanup
    py::gil_scoped_acquire acquire;

    // Clean up
    action_sequence.clear();
    env.reset();
  }

protected:
  std::unique_ptr<MettaGrid> env;
  std::vector<py::array_t<int>> action_sequence;
  py::array_t<int, py::array::c_style> actions_buffer;  // Store actions buffer for mutation
  size_t num_agents;
  size_t iteration_counter;
  std::string setup_error;
};

// Matching Python test_step_performance_no_reset
BENCHMARK_F(MettaGridBenchmark, Step)(benchmark::State& state) {
  // Check for setup errors
  if (!setup_error.empty()) {
    state.SkipWithError(setup_error.c_str());
    return;
  }

  // Acquire GIL for the entire benchmark
  py::gil_scoped_acquire acquire;

  // Benchmark loop
  for (auto _ : state) {
    // Get the next action from the pre-generated sequence
    const auto& action_array = action_sequence[iteration_counter % action_sequence.size()];
    iteration_counter++;

    // Copy actions into the actions buffer
    auto* actions_ptr = static_cast<int*>(actions_buffer.mutable_unchecked<1>().mutable_data(0));
    auto* action_array_ptr = static_cast<int*>(action_array.request().ptr);
    std::copy(action_array_ptr, action_array_ptr + num_agents, actions_ptr);

    // Update buffers with new actions (set_buffers copies, so we need to call it)
    auto observations = env->observations();
    auto terminals = env->terminals();
    auto truncations = env->truncations();
    auto rewards = env->rewards();
    env->set_buffers(observations, terminals, truncations, rewards, actions_buffer);

    // Perform the step (no arguments - uses actions from buffer)
    env->step();

    // Note: Intentionally ignoring termination states to measure pure step performance,
    // matching the Python implementation
  }

  // Only set counters if iterations were actually performed
  if (state.iterations() > 0) {
    // Report steps/second as custom counters
    // Use explicit string construction to avoid potential ASan issues with string literals
    state.counters[std::string("env_rate")] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
    state.counters[std::string("agent_rate")] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(num_agents), benchmark::Counter::kIsRate);
  }
}

// Custom main that properly initializes Python
int main(int argc, char** argv) {
  // Initialize Python interpreter BEFORE benchmark initialization
  // Store it in a global to ensure it stays alive
  g_python_guard = std::make_unique<py::scoped_interpreter>();

  // Now initialize benchmark framework
  ::benchmark::Initialize(&argc, argv);

  // Run benchmarks
  ::benchmark::RunSpecifiedBenchmarks();

  // Shutdown benchmark framework
  ::benchmark::Shutdown();

  // Clean up Python interpreter
  g_python_guard.reset();

  return 0;
}
