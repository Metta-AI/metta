#include <benchmark/benchmark.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <vector>

#include "actions/attack.hpp"
#include "actions/change_glyph.hpp"
#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/wall.hpp"

namespace py = pybind11;

// TODO: Currently this benchmark requires Python/pybind11 because the MettaGrid
// API is tightly coupled with Python types (py::dict, py::list, py::array_t).
//
// The goal is to refactor MettaGrid to have a pure C++ core with a thin Python
// wrapper layer. When that refactoring is complete, we should be able to:
// 1. Remove all pybind11 includes from this benchmark
// 2. Remove the Python interpreter initialization
// 3. Work directly with C++ types (std::vector, std::map, etc.)
//
// We'll know we've succeeded when this file has zero references to pybind11!

// Helper functions for creating configuration and map
GameConfig CreateBenchmarkConfig(size_t num_agents) {
  std::vector<std::string> inventory_item_names = {"ore", "heart"};

  std::shared_ptr<ActionConfig> action_cfg = std::make_shared<ActionConfig>(
      std::map<InventoryItem, InventoryQuantity>(), std::map<InventoryItem, InventoryQuantity>());

  std::shared_ptr<AttackActionConfig> attack_cfg =
      std::make_shared<AttackActionConfig>(std::map<InventoryItem, InventoryQuantity>(),
                                           std::map<InventoryItem, InventoryQuantity>(),
                                           std::map<InventoryItem, InventoryQuantity>());

  std::shared_ptr<ChangeGlyphActionConfig> change_glyph_cfg = std::make_shared<ChangeGlyphActionConfig>(
      std::map<InventoryItem, InventoryQuantity>(), std::map<InventoryItem, InventoryQuantity>(), 4);

  std::map<std::string, std::shared_ptr<ActionConfig>> actions_cfg;

  actions_cfg["noop"] = action_cfg;
  actions_cfg["move"] = action_cfg;
  actions_cfg["rotate"] = action_cfg;
  actions_cfg["attack"] = attack_cfg;
  actions_cfg["swap"] = action_cfg;
  actions_cfg["put_items"] = action_cfg;
  actions_cfg["get_items"] = action_cfg;
  actions_cfg["change_color"] = action_cfg;
  actions_cfg["change_glyph"] = change_glyph_cfg;

  std::map<std::string, std::shared_ptr<GridObjectConfig>> objects_cfg;

  objects_cfg["wall"] = std::make_shared<WallConfig>(1, "wall", false);
  objects_cfg["agent.team1"] = std::make_shared<AgentConfig>(0,
                                                             "agent",
                                                             0,
                                                             "team1",
                                                             0,
                                                             0.0f,
                                                             std::map<InventoryItem, InventoryQuantity>(),
                                                             std::map<InventoryItem, RewardType>(),
                                                             std::map<InventoryItem, RewardType>(),
                                                             std::map<std::string, RewardType>(),
                                                             std::map<std::string, RewardType>(),
                                                             0.0f);
  objects_cfg["agent.team2"] = std::make_shared<AgentConfig>(0,
                                                             "agent",
                                                             1,
                                                             "team2",
                                                             0,
                                                             0.0f,
                                                             std::map<InventoryItem, InventoryQuantity>(),
                                                             std::map<InventoryItem, RewardType>(),
                                                             std::map<InventoryItem, RewardType>(),
                                                             std::map<std::string, RewardType>(),
                                                             std::map<std::string, RewardType>(),
                                                             0.0f);

  // Create default global observation config
  GlobalObsConfig global_obs_config;
  global_obs_config.episode_completion_pct = true;
  global_obs_config.last_action = true;
  global_obs_config.last_reward = true;
  global_obs_config.resource_rewards = true;

  return GameConfig(
      num_agents, 10000, false, 11, 11, inventory_item_names, 100, global_obs_config, actions_cfg, objects_cfg);
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
py::array_t<int> GenerateValidRandomActions(MettaGrid* env, size_t num_agents, std::mt19937* gen) {
  // Get the maximum argument values for each action type
  py::list max_args = env->max_action_args();
  size_t num_actions = py::len(env->action_names());

  // Create actions array
  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(num_agents), 2};
  py::array_t<int> actions(shape);
  auto actions_ptr = static_cast<int*>(actions.request().ptr);

  // Initialize distributions
  std::uniform_int_distribution<> action_dist(0, static_cast<int>(num_actions) - 1);

  for (size_t i = 0; i < num_agents; ++i) {
    // Choose random action type
    int action_type = action_dist(*gen);

    // Get max allowed argument for this action type
    int max_arg = py::cast<int>(max_args[static_cast<size_t>(action_type)]);

    // Choose random valid argument (0 to max_arg inclusive)
    int action_arg = 0;
    if (max_arg > 0) {
      std::uniform_int_distribution<> arg_dist(0, max_arg);
      action_arg = arg_dist(*gen);
    }

    // Set the action values
    actions_ptr[i * 2] = action_type;
    actions_ptr[i * 2 + 1] = action_arg;
  }

  return actions;
}

// Pre-generate a sequence of actions for the benchmark
std::vector<py::array_t<int>> PreGenerateActionSequence(MettaGrid* env, size_t num_agents, size_t sequence_length) {
  std::vector<py::array_t<int>> action_sequence;
  action_sequence.reserve(static_cast<size_t>(sequence_length));

  // Use a deterministic seed for reproducible benchmarks
  std::mt19937 gen(42);

  for (size_t i = 0; i < sequence_length; ++i) {
    action_sequence.push_back(GenerateValidRandomActions(env, num_agents, &gen));
  }

  return action_sequence;
}

// Matching Python test_step_performance_no_reset
static void BM_MettaGridStep(benchmark::State& state) {  // NOLINT(runtime/references)
  // Ensure we have the GIL for all Python operations
  py::gil_scoped_acquire acquire;

  // Setup with default 4 agents (matching Python benchmark config)
  size_t num_agents = 4;
  auto cfg = CreateBenchmarkConfig(num_agents);
  auto map = CreateDefaultMap(2);

  auto env = std::make_unique<MettaGrid>(cfg, map, 42);
  env->reset();

  // Verify agent count
  if (env->num_agents() != num_agents) {
    state.SkipWithError("Agent count mismatch");
    return;
  }

  // Pre-generate action sequence matching Python benchmark
  // Python uses: iterations = 1000, rounds = 20, total = 20000
  const int iterations = 1000;
  const int rounds = 20;
  const int total_iterations = iterations * rounds;

  auto action_sequence = PreGenerateActionSequence(env.get(), num_agents, total_iterations);

  // Counter to track which action to use
  size_t iteration_counter = 0;

  // Benchmark loop
  for (auto _ : state) {
    // Get the next action from the pre-generated sequence
    const auto& actions = action_sequence[iteration_counter % action_sequence.size()];
    iteration_counter++;

    // Perform the step
    auto result = env->step(actions);
    benchmark::DoNotOptimize(result);

    // Note: Intentionally ignoring termination states to measure pure step performance,
    // matching the Python implementation
  }

  // Report steps/second as custom counters
  state.counters["env_rate"] = benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.counters["agent_rate"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(num_agents), benchmark::Counter::kIsRate);
}

// Custom main that properly initializes Python
int main(int argc, char** argv) {
  // Initialize Python interpreter first, before any benchmark registration
  py::scoped_interpreter guard{};

  // Initialize benchmark framework
  ::benchmark::Initialize(&argc, argv);

  // Register benchmarks after Python is initialized
  // Use Threads(1) to ensure single-threaded execution for Python GIL safety
  ::benchmark::RegisterBenchmark("BM_MettaGridStep", BM_MettaGridStep)
      ->Unit(benchmark::kMillisecond)
      ->Threads(1);

  // Run benchmarks
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
