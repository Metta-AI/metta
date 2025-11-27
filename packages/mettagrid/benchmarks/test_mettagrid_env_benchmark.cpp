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
#include "actions/move_config.hpp"   // For MoveActionConfig
#include "actions/resource_mod.hpp"  // For ResourceModConfig
#include "bindings/mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/agent_config.hpp"
#include "objects/wall.hpp"

namespace py = pybind11;

// Hold a Python interpreter for process lifetime
static std::unique_ptr<py::scoped_interpreter> g_python_guard;

// Initialize Python very early to avoid any static-init use of NumPy C-API
struct PythonBootstrap {
  PythonBootstrap() {
    try {
      g_python_guard = std::make_unique<py::scoped_interpreter>();
      // Import numpy so py::array_t is safe to construct later
      py::module_::import("numpy");
    } catch (const std::exception& e) {
      fprintf(stderr, "[mettagrid-bench] early python bootstrap failed: %s\n", e.what());
    }
  }
};
static PythonBootstrap g_bootstrap __attribute__((init_priority(101)));

// Helper: construct a minimal GameConfig for the benchmark
GameConfig CreateBenchmarkConfig(size_t num_agents) {
  std::vector<std::string> resource_names = {"ore", "heart"};

  auto action_cfg = std::make_shared<ActionConfig>(std::unordered_map<InventoryItem, InventoryQuantity>{},
                                                   std::unordered_map<InventoryItem, InventoryProbability>{});
  auto move_cfg = std::make_shared<MoveActionConfig>(std::vector<std::string>{"north", "south", "east", "west"},
                                                     std::unordered_map<InventoryItem, InventoryQuantity>{},
                                                     std::unordered_map<InventoryItem, InventoryProbability>{});
  auto attack_cfg = std::make_shared<AttackActionConfig>(std::unordered_map<InventoryItem, InventoryQuantity>{},
                                                         std::unordered_map<InventoryItem, InventoryProbability>{},
                                                         std::unordered_map<InventoryItem, InventoryQuantity>{});
  auto change_vibe_cfg =
      std::make_shared<ChangeVibeActionConfig>(std::unordered_map<InventoryItem, InventoryQuantity>{},
                                               std::unordered_map<InventoryItem, InventoryProbability>{},
                                               4);
  // Create proper ResourceModConfig with all required fields
  auto resource_mod_cfg = std::make_shared<ResourceModConfig>(
      std::unordered_map<InventoryItem, InventoryQuantity>{},     // required_resources
      std::unordered_map<InventoryItem, InventoryProbability>{},  // consumed_resources
      std::unordered_map<InventoryItem, InventoryProbability>{},  // modifies
      0,                                                          // agent_radius
      false);                                                     // scales

  std::unordered_map<std::string, std::shared_ptr<ActionConfig>> actions_cfg;
  actions_cfg["noop"] = action_cfg;
  actions_cfg["move"] = move_cfg;
  actions_cfg["rotate"] = action_cfg;
  actions_cfg["attack"] = attack_cfg;
  actions_cfg["change_vibe"] = change_vibe_cfg;
  actions_cfg["resource_mod"] = resource_mod_cfg;

  std::unordered_map<std::string, std::shared_ptr<GridObjectConfig>> objects_cfg;
  objects_cfg["wall"] = std::make_shared<WallConfig>(1, "wall", false);
  objects_cfg["agent.team1"] = std::make_shared<AgentConfig>(0, "agent", 0, "team1");
  objects_cfg["agent.team2"] = std::make_shared<AgentConfig>(0, "agent", 1, "team2");

  GlobalObsConfig global_obs_config;
  global_obs_config.episode_completion_pct = true;
  global_obs_config.last_action = true;
  global_obs_config.last_reward = true;

  // Minimal observation feature id map needed by C++ core - CRITICAL!
  std::unordered_map<std::string, ObservationType> feature_ids;
  ObservationType fid = 1;
  auto add = [&](const char* name) { feature_ids.emplace(std::string(name), fid++); };
  add("agent:group");
  add("agent:frozen");
  add("agent:orientation");
  add("agent:reserved_for_future_use");
  add("converting");
  add("swappable");
  add("episode_completion_pct");
  add("last_action");
  add("last_action_arg");
  add("last_reward");
  add("vibe");
  add("agent:visitation_counts");
  add("agent:compass");
  add("tag");
  add("cooldown_remaining");
  add("clipped");
  add("remaining_uses");
  add("inv:ore");
  add("inv:heart");

  std::unordered_map<int, std::string> tag_id_map;

  return GameConfig(/*num_agents*/ num_agents,
                    /*max_steps*/ 10000,
                    /*episode_truncates*/ false,
                    /*obs_width*/ 11,
                    /*obs_height*/ 11,
                    /*resource_names*/ resource_names,
                    /*vibe_names*/ std::vector<std::string>{},
                    /*num_observation_tokens*/ 100,
                    /*global_obs*/ global_obs_config,
                    /*feature_ids*/ feature_ids,
                    /*actions*/ actions_cfg,
                    /*objects*/ objects_cfg,
                    /*resource_loss_prob*/ 0.0f,
                    /*tag_id_map*/ tag_id_map,
                    /*protocol_details_obs*/ false,
                    /*reward_estimates*/ std::unordered_map<std::string, float>{},
                    /*inventory_regen_interval*/ 0,
                    /*clipper*/ nullptr);
}

py::list CreateDefaultMap(size_t num_agents_per_team = 2) {
  py::list map;
  const int width = 32;
  const int height = 32;
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

class MettaGridBenchmark : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State&) override {
    if (!g_python_guard) {
      setup_error = "Python interpreter not initialized";
      return;
    }
    py::gil_scoped_acquire gil;

    num_agents = 4;
    auto cfg = CreateBenchmarkConfig(num_agents);
    auto map = CreateDefaultMap(2);
    try {
      env = std::make_unique<MettaGrid>(cfg, map, 42);
    } catch (const std::exception& e) {
      setup_error = std::string("Failed to create environment: ") + e.what();
      return;
    } catch (...) {
      setup_error = "Failed to create environment: unknown exception";
      return;
    }

    const size_t num_tokens = cfg.num_observation_tokens;
    std::vector<py::ssize_t> obs_shape = {
        static_cast<py::ssize_t>(num_agents), static_cast<py::ssize_t>(num_tokens), 3};
    observations_buffer = py::array_t<uint8_t, py::array::c_style>(obs_shape);
    terminals_buffer = py::array_t<bool, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    truncations_buffer = py::array_t<bool, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    rewards_buffer = py::array_t<float, py::array::c_style>(static_cast<py::ssize_t>(num_agents));
    actions_buffer = py::array_t<int, py::array::c_style>(static_cast<py::ssize_t>(num_agents));

    // zero actions
    std::fill(static_cast<int*>(actions_buffer.request().ptr),
              static_cast<int*>(actions_buffer.request().ptr) + actions_buffer.size(),
              0);

    env->set_buffers(observations_buffer, terminals_buffer, truncations_buffer, rewards_buffer, actions_buffer);

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
    py::gil_scoped_acquire gil;

    // Clean up
    action_sequence.clear();
    env.reset();
  }

protected:
  std::unique_ptr<MettaGrid> env;
  std::vector<py::array_t<int>> action_sequence;
  py::array_t<uint8_t, py::array::c_style> observations_buffer;
  py::array_t<bool, py::array::c_style> terminals_buffer;
  py::array_t<bool, py::array::c_style> truncations_buffer;
  py::array_t<float, py::array::c_style> rewards_buffer;
  py::array_t<int, py::array::c_style> actions_buffer;
  size_t num_agents{};
  size_t iteration_counter;
  std::string setup_error;
};

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

int main(int argc, char** argv) {
  // Python should already be initialized by PythonBootstrap static initializer
  if (!Py_IsInitialized()) {
    fprintf(stderr, "[mettagrid-bench] WARNING: Python not initialized by bootstrap, initializing now\n");
    try {
      g_python_guard = std::make_unique<py::scoped_interpreter>();
    } catch (const std::exception& e) {
      fprintf(stderr, "[mettagrid-bench] failed to start interpreter: %s\n", e.what());
      return 1;
    }
  }
  {
    py::gil_scoped_acquire gil;
    try {
      py::module_::import("numpy");
    } catch (const std::exception& e) {
      fprintf(stderr, "Failed to import numpy: %s\n", e.what());
      return 1;
    }
  }
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}