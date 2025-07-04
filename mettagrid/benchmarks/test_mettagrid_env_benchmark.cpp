#include <benchmark/benchmark.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <vector>

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
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
py::dict CreateBenchmarkConfig(int num_agents) {
  py::dict game_cfg;

  // Basic game configuration
  game_cfg["num_agents"] = num_agents;
  game_cfg["max_steps"] = 10000;
  game_cfg["obs_width"] = 11;
  game_cfg["obs_height"] = 11;
  game_cfg["num_observation_tokens"] = 100;

  // Import mettagrid_c module to get Python configuration classes
  py::module_ mettagrid_c = py::module_::import("metta.mettagrid.mettagrid_c");

  // Inventory item names configuration
  py::list inventory_item_names;
  inventory_item_names.append("ore");
  inventory_item_names.append("heart");
  game_cfg["inventory_item_names"] = inventory_item_names;

  // Actions configuration
  py::object ActionConfig = mettagrid_c.attr("ActionConfig");
  py::object AttackActionConfig = mettagrid_c.attr("AttackActionConfig");

  py::dict actions_cfg;
  py::object action_cfg = ActionConfig(true, py::dict(), py::dict());
  py::object attack_cfg = AttackActionConfig(true, py::dict(), py::dict(), py::dict());  // NOLINT(readability/fn_size)

  actions_cfg["noop"] = action_cfg;
  actions_cfg["move"] = action_cfg;
  actions_cfg["rotate"] = action_cfg;
  actions_cfg["attack"] = attack_cfg;
  actions_cfg["swap"] = action_cfg;
  actions_cfg["put_items"] = action_cfg;
  actions_cfg["get_items"] = action_cfg;
  actions_cfg["change_color"] = action_cfg;

  game_cfg["actions"] = actions_cfg;

  // Create Python AgentConfig objects
  py::object AgentConfig = mettagrid_c.attr("AgentConfig");
  py::object WallConfig = mettagrid_c.attr("WallConfig");

  py::object agent_cfg1 = AgentConfig(0,           // type_id
                                      "agent",     // type_name
                                      0,           // group_id
                                      "team1",     // group_name
                                      0,           // freeze_duration
                                      0.0f,        // action_failure_penalty
                                      py::dict(),  // max_items_per_type
                                      py::dict(),  // resource_rewards
                                      py::dict(),  // resource_reward_max
                                      0.0f);       // group_reward_pct

  py::object agent_cfg2 = AgentConfig(0,           // type_id
                                      "agent",     // type_name
                                      1,           // group_id
                                      "team2",     // group_name
                                      0,           // freeze_duration
                                      0.0f,        // action_failure_penalty
                                      py::dict(),  // max_items_per_type
                                      py::dict(),  // resource_rewards
                                      py::dict(),  // resource_reward_max
                                      0.0f);       // group_reward_pct

  // Objects configuration
  py::dict objects_cfg;

  py::object wall_cfg = WallConfig(1, "wall", false);

  objects_cfg["wall"] = wall_cfg;
  objects_cfg["agent.team1"] = agent_cfg1;
  objects_cfg["agent.team2"] = agent_cfg2;

  game_cfg["objects"] = objects_cfg;

  return game_cfg;
}

py::list CreateDefaultMap(int num_agents_per_team = 2) {
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
  int agents_placed = 0;
  std::vector<std::pair<int, int>> positions = {{8, 8},   {8, 24},  {24, 8},  {24, 24}, {16, 8},  {16, 24}, {8, 16},
                                                {24, 16}, {12, 12}, {12, 20}, {20, 12}, {20, 20}, {16, 16}, {16, 12},
                                                {16, 20}, {12, 16}, {20, 16}, {10, 10}, {10, 22}, {22, 10}, {22, 22},
                                                {14, 14}, {14, 18}, {18, 14}, {18, 18}};

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
py::array_t<int> GenerateValidRandomActions(MettaGrid* env, int num_agents, std::mt19937* gen) {
  // Get the maximum argument values for each action type
  py::list max_args = env->max_action_args();
  int num_actions = py::len(env->action_names());

  // Create actions array
  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(num_agents), 2};
  py::array_t<int> actions(shape);
  auto actions_ptr = static_cast<int*>(actions.request().ptr);

  // Initialize distributions
  std::uniform_int_distribution<> action_dist(0, num_actions - 1);

  for (int i = 0; i < num_agents; ++i) {
    // Choose random action type
    int action_type = action_dist(*gen);

    // Get max allowed argument for this action type
    int max_arg = py::cast<int>(max_args[action_type]);

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
std::vector<py::array_t<int>> PreGenerateActionSequence(MettaGrid* env, int num_agents, int sequence_length) {
  std::vector<py::array_t<int>> action_sequence;
  action_sequence.reserve(sequence_length);

  // Use a deterministic seed for reproducible benchmarks
  std::mt19937 gen(42);

  for (int i = 0; i < sequence_length; ++i) {
    action_sequence.push_back(GenerateValidRandomActions(env, num_agents, &gen));
  }

  return action_sequence;
}

// Matching Python test_step_performance_no_reset
static void BM_MettaGridStep(benchmark::State& state) {  // NOLINT(runtime/references)
  // Setup with default 4 agents (matching Python benchmark config)
  int num_agents = 4;
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
  state.counters["env_rate"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["agent_rate"] = benchmark::Counter(state.iterations() * num_agents, benchmark::Counter::kIsRate);
}

// Register benchmarks to match Python tests
BENCHMARK(BM_MettaGridStep)->Unit(benchmark::kMillisecond);

// Custom main that properly initializes Python
int main(int argc, char** argv) {
  py::scoped_interpreter guard{};

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
