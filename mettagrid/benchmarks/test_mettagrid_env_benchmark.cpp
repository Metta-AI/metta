#include <benchmark/benchmark.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <vector>

#include "mettagrid_c.hpp"

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
  py::dict cfg;
  py::dict game_cfg;

  // Basic game configuration
  game_cfg["num_agents"] = num_agents;  // Now dynamically set based on actual agents
  game_cfg["max_steps"] = 10000;
  game_cfg["obs_width"] = 11;
  game_cfg["obs_height"] = 11;
  game_cfg["use_observation_tokens"] = false;

  // Actions configuration
  py::dict actions_cfg;
  py::dict noop_cfg, move_cfg, rotate_cfg, attack_cfg, swap_cfg, put_cfg, get_cfg, change_color_cfg;

  noop_cfg["enabled"] = true;
  move_cfg["enabled"] = true;
  rotate_cfg["enabled"] = true;
  attack_cfg["enabled"] = true;
  swap_cfg["enabled"] = true;
  put_cfg["enabled"] = true;
  get_cfg["enabled"] = true;
  change_color_cfg["enabled"] = true;

  actions_cfg["noop"] = noop_cfg;
  actions_cfg["move"] = move_cfg;
  actions_cfg["rotate"] = rotate_cfg;
  actions_cfg["attack"] = attack_cfg;
  actions_cfg["swap"] = swap_cfg;
  actions_cfg["put_items"] = put_cfg;
  actions_cfg["get_items"] = get_cfg;
  actions_cfg["change_color"] = change_color_cfg;

  game_cfg["actions"] = actions_cfg;

  // Groups configuration
  py::dict groups;
  py::dict group1, group2;

  group1["id"] = 0;
  group1["group_reward_pct"] = 0.0f;
  py::dict group1_props;
  group1["props"] = group1_props;

  group2["id"] = 1;
  group2["group_reward_pct"] = 0.0f;
  py::dict group2_props;
  group2["props"] = group2_props;

  groups["team1"] = group1;
  groups["team2"] = group2;

  game_cfg["groups"] = groups;

  // Objects configuration
  py::dict objects_cfg;
  py::dict wall_cfg, block_cfg, agent_cfg, mine_cfg, generator_cfg, altar_cfg;

  objects_cfg["wall"] = wall_cfg;
  objects_cfg["block"] = block_cfg;
  objects_cfg["mine.red"] = mine_cfg;
  objects_cfg["generator.red"] = generator_cfg;
  objects_cfg["altar"] = altar_cfg;

  game_cfg["objects"] = objects_cfg;
  game_cfg["agent"] = agent_cfg;

  cfg["game"] = game_cfg;

  return cfg;
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

  // Place agents symmetrically - ensure we have enough positions
  int agents_placed = 0;
  std::vector<std::pair<int, int>> positions = {// Corners of inner square
                                                {8, 8},
                                                {8, 24},
                                                {24, 8},
                                                {24, 24},
                                                // Midpoints of edges
                                                {16, 8},
                                                {16, 24},
                                                {8, 16},
                                                {24, 16},
                                                // Inner square
                                                {12, 12},
                                                {12, 20},
                                                {20, 12},
                                                {20, 20},
                                                // Center and additional positions
                                                {16, 16},
                                                {16, 12},
                                                {16, 20},
                                                {12, 16},
                                                {20, 16},
                                                // More positions if needed
                                                {10, 10},
                                                {10, 22},
                                                {22, 10},
                                                {22, 22},
                                                {14, 14},
                                                {14, 18},
                                                {18, 14},
                                                {18, 18}};

  for (size_t i = 0; i < positions.size() && agents_placed < num_agents_per_team * 2; ++i) {
    auto [r, c] = positions[i];
    std::string team = (agents_placed % 2 == 0) ? "agent.team1" : "agent.team2";
    // Need to cast the row to py::list before accessing column
    py::list row = map[r].cast<py::list>();
    row[c] = team;
    agents_placed++;
  }

  return map;
}

// Static benchmark for step performance
static void BM_MettaGridStep(benchmark::State& state) {
  // Note: Python interpreter must be initialized before this runs

  // Setup with default 4 agents (2 per team)
  int num_agents = 4;
  auto cfg = CreateBenchmarkConfig(num_agents);
  auto map = CreateDefaultMap(2);  // 2 agents per team

  // Create initial environment
  auto env = std::make_unique<MettaGrid>(cfg, map);
  env->reset();  // Initial reset before any steps

  // Verify we have the expected number of agents
  auto actual_num_agents = env->num_agents();
  if (actual_num_agents != num_agents) {
    state.SkipWithError("Agent count mismatch");
    return;
  }

  // Create shape vector for array_t constructor
  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(num_agents), 2};
  py::array_t<int> actions(shape);

  auto actions_ptr = static_cast<int*>(actions.request().ptr);

  // Initialize with random actions
  std::mt19937 gen(42);
  std::uniform_int_distribution<> action_dist(0, py::len(env->action_names()) - 1);
  std::uniform_int_distribution<> arg_dist(0, 3);

  for (size_t i = 0; i < num_agents; ++i) {
    actions_ptr[i * 2] = action_dist(gen);
    actions_ptr[i * 2 + 1] = arg_dist(gen);
  }

  // Benchmark loop - only measure step performance
  // Assume episode won't terminate within benchmark iterations
  for (auto _ : state) {
    auto result = env->step(actions);
    benchmark::DoNotOptimize(result);
  }

  // Set performance counters using only custom counters
  state.counters["num_agents"] = static_cast<double>(num_agents);
  state.counters["env_steps_per_sec"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["agent_steps_per_sec"] =
      benchmark::Counter(state.iterations() * num_agents, benchmark::Counter::kIsRate);
}

// Static benchmark for reset performance
static void BM_MettaGridReset(benchmark::State& state) {
  // Setup configuration with default 4 agents
  auto cfg = CreateBenchmarkConfig(4);
  auto map = CreateDefaultMap(2);

  // Benchmark loop - recreate environment each time since reset()
  // can only be called on fresh environments
  for (auto _ : state) {
    auto env = std::make_unique<MettaGrid>(cfg, map);
    auto result = env->reset();
    benchmark::DoNotOptimize(result);
  }

  // Set performance counter using only custom counter
  state.counters["resets_per_sec"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

// Remove the automatic items_per_second from BM_EnvironmentCreation
static void BM_EnvironmentCreation(benchmark::State& state) {
  auto cfg = CreateBenchmarkConfig(4);
  auto map = CreateDefaultMap(2);

  for (auto _ : state) {
    auto env = std::make_unique<MettaGrid>(cfg, map);
    benchmark::DoNotOptimize(env);
  }

  // Use custom counter instead of SetItemsProcessed
  state.counters["envs_created_per_sec"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

// Parameterized benchmark for different agent counts
static void BM_MettaGridStepAgentScaling(benchmark::State& state) {
  int num_agents_per_team = state.range(0);
  int total_agents = num_agents_per_team * 2;

  auto cfg = CreateBenchmarkConfig(total_agents);
  auto map = CreateDefaultMap(num_agents_per_team);
  auto env = std::make_unique<MettaGrid>(cfg, map);
  env->reset();  // Initial reset before any steps

  // Verify agent count
  if (env->num_agents() != total_agents) {
    state.SkipWithError("Agent count mismatch in scaling test");
    return;
  }

  // Create actions array
  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(total_agents), 2};
  py::array_t<int> actions(shape);
  auto actions_ptr = static_cast<int*>(actions.request().ptr);

  // Random action generation
  std::mt19937 gen(42);
  std::uniform_int_distribution<> action_dist(0, py::len(env->action_names()) - 1);
  std::uniform_int_distribution<> arg_dist(0, 3);

  for (int i = 0; i < total_agents; ++i) {
    actions_ptr[i * 2] = action_dist(gen);
    actions_ptr[i * 2 + 1] = arg_dist(gen);
  }

  // Benchmark loop - only measure step performance
  for (auto _ : state) {
    auto result = env->step(actions);
    benchmark::DoNotOptimize(result);
  }

  // Set performance counters using only custom counters
  state.counters["total_agents"] = static_cast<double>(total_agents);
  state.counters["agents_per_team"] = static_cast<double>(num_agents_per_team);
  state.counters["env_steps_per_sec"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
  state.counters["agent_steps_per_sec"] =
      benchmark::Counter(state.iterations() * total_agents, benchmark::Counter::kIsRate);
}

// Register benchmarks
BENCHMARK(BM_MettaGridStep);
BENCHMARK(BM_MettaGridReset);
BENCHMARK(BM_EnvironmentCreation);

// Register parameterized benchmark with different agent counts
BENCHMARK(BM_MettaGridStepAgentScaling)
    ->Arg(1)  // 2 total agents
    ->Arg(2)  // 4 total agents
    ->Arg(4)  // 8 total agents
    ->Arg(8)  // 16 total agents
    ->Unit(benchmark::kMicrosecond);

// Custom main that properly initializes Python
int main(int argc, char** argv) {
  // Initialize Python interpreter
  py::scoped_interpreter guard{};

  // Run benchmarks
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}