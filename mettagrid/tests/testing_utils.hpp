#ifndef METTAGRID_TESTS_TESTING_UTILS_HPP_
#define METTAGRID_TESTS_TESTING_UTILS_HPP_

#include <pybind11/numpy.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#include "mettagrid_c.hpp"

namespace testing_utils {

// Test-friendly subclass that exposes private members for testing
// This requires changing private members to protected in mettagrid_c.hpp
class TestMettaGrid : public MettaGrid {
public:
  TestMettaGrid(py::dict env_cfg, py::list map) : MettaGrid(env_cfg, map) {}

  // Expose protected members for testing
  bool get_reward_decay_enabled() const {
    return _reward_decay_enabled;
  }
  float get_reward_decay_factor() const {
    return _reward_decay_factor;
  }
  float get_min_reward_multiplier() const {
    return _min_reward_multiplier;
  }

  // You can add more testing accessors here as needed
  // For example:
  // const std::vector<Agent*>& get_agents() const { return _agents; }
  // Grid* get_grid() const { return _grid.get(); }
};

// Structure to hold all buffer pointers for easier management
struct GridBuffers {
  ObservationType* observations = nullptr;
  TerminalType* terminals = nullptr;
  TruncationType* truncations = nullptr;
  RewardType* rewards = nullptr;

  // Size information for reference
  size_t obs_size = 0;
  size_t terminals_size = 0;
  size_t truncations_size = 0;
  size_t rewards_size = 0;
};

// Create a simple test map and config for MettaGrid
inline py::list create_test_map() {
  // Create a simple 10x10 map with agents at opposite corners
  py::list map;
  for (int r = 0; r < 10; r++) {
    py::list row;
    for (int c = 0; c < 10; c++) {
      if (r == 0 && c == 0) {
        row.append("agent.red");
      } else if (r == 9 && c == 9) {
        row.append("agent.blue");
      } else {
        row.append("empty");
      }
    }
    map.append(row);
  }
  return map;
}

inline py::dict create_test_config() {
  // Create minimal config for testing
  py::dict config;
  py::dict game;

  // Basic game settings
  game["num_agents"] = 2;
  game["max_steps"] = 100;
  game["obs_width"] = 5;
  game["obs_height"] = 5;

  // Agent configuration
  py::dict agent;
  agent["health"] = 100;
  agent["max_health"] = 100;
  agent["energy"] = 100;
  agent["max_energy"] = 100;
  game["agent"] = agent;

  // Groups configuration
  py::dict groups;
  py::dict red_group;
  red_group["id"] = 0;
  red_group["group_reward_pct"] = 0.0f;
  py::dict red_props;
  red_props["color"] = 1;
  red_group["props"] = red_props;
  groups["red"] = red_group;

  py::dict blue_group;
  blue_group["id"] = 1;
  blue_group["group_reward_pct"] = 0.0f;
  py::dict blue_props;
  blue_props["color"] = 2;
  blue_group["props"] = blue_props;
  groups["blue"] = blue_group;
  game["groups"] = groups;

  // Actions configuration - enable minimal set
  py::dict actions;
  py::dict noop_action;
  noop_action["enabled"] = true;
  noop_action["priority"] = 0;
  actions["noop"] = noop_action;

  py::dict move_action;
  move_action["enabled"] = true;
  move_action["priority"] = 1;
  actions["move"] = move_action;

  py::dict attack_action;
  attack_action["enabled"] = false;  // Disable for simple testing
  actions["attack"] = attack_action;

  py::dict put_items_action;
  put_items_action["enabled"] = false;
  actions["put_items"] = put_items_action;

  py::dict get_items_action;
  get_items_action["enabled"] = false;
  actions["get_items"] = get_items_action;

  py::dict rotate_action;
  rotate_action["enabled"] = false;
  actions["rotate"] = rotate_action;

  py::dict swap_action;
  swap_action["enabled"] = false;
  actions["swap"] = swap_action;

  py::dict change_color_action;
  change_color_action["enabled"] = false;
  actions["change_color"] = change_color_action;

  game["actions"] = actions;

  // Objects configuration (minimal)
  py::dict objects;
  py::dict wall;
  wall["health"] = 100;
  objects["wall"] = wall;
  game["objects"] = objects;

  config["game"] = game;
  return config;
}

// Create a standard test grid with sensible defaults
inline std::unique_ptr<MettaGrid> create_test_grid(uint32_t map_width = 10,
                                                   uint32_t map_height = 10,
                                                   uint32_t num_agents = 2,
                                                   uint32_t max_timestep = 100,
                                                   uint16_t obs_width = 5,
                                                   uint16_t obs_height = 5) {
  try {
    py::dict config = create_test_config();
    py::list map = create_test_map();

    // Update config with provided parameters
    config["game"]["max_steps"] = max_timestep;
    config["game"]["obs_width"] = obs_width;
    config["game"]["obs_height"] = obs_height;
    config["game"]["num_agents"] = num_agents;

    // Create MettaGrid with config and map
    auto grid = std::make_unique<MettaGrid>(config, map);

    return grid;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create test grid: " << e.what() << std::endl;
    throw;
  }
}

// Create a TestMettaGrid for advanced testing
inline std::unique_ptr<TestMettaGrid> create_test_mettagrid(uint32_t map_width = 10,
                                                            uint32_t map_height = 10,
                                                            uint32_t num_agents = 2,
                                                            uint32_t max_timestep = 100,
                                                            uint16_t obs_width = 5,
                                                            uint16_t obs_height = 5) {
  try {
    py::dict config = create_test_config();
    py::list map = create_test_map();

    // Update config with provided parameters
    config["game"]["max_steps"] = max_timestep;
    config["game"]["obs_width"] = obs_width;
    config["game"]["obs_height"] = obs_height;
    config["game"]["num_agents"] = num_agents;

    // Create TestMettaGrid with config and map
    auto grid = std::make_unique<TestMettaGrid>(config, map);

    return grid;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create test TestMettaGrid: " << e.what() << std::endl;
    throw;
  }
}

// Create a flat action array for testing (2 values per agent: action_type, action_arg)
inline ActionType* create_action_array(uint32_t num_agents,
                                       ActionType action_type = 0,  // Default to first action (likely Noop)
                                       ActionType action_arg = 0) {
  if (num_agents == 0) {
    throw std::invalid_argument("Number of agents must be greater than 0");
  }

  // Allocate flat array: 2 values per agent (action_type, action_arg)
  ActionType* actions = new ActionType[num_agents * 2];

  for (uint32_t i = 0; i < num_agents; ++i) {
    size_t base_idx = i * 2;
    actions[base_idx] = action_type;
    actions[base_idx + 1] = action_arg;
  }

  return actions;
}

// Clean up action array
inline void delete_action_array(ActionType* actions) {
  delete[] actions;
}

// Allocate buffers for a grid and connect them - works with any MettaGrid-derived class
template <typename GridType>
GridBuffers* allocate_grid_buffers(GridType* grid) {
  if (!grid) {
    throw std::invalid_argument("Cannot allocate buffers for null grid");
  }

  auto buffers = std::make_unique<GridBuffers>();

  try {
    // Get basic dimensions from grid
    uint32_t num_agents = grid->num_agents();
    uint32_t obs_height = grid->obs_height;
    uint32_t obs_width = grid->obs_width;

    // Calculate buffer sizes
    auto feature_norms = grid->feature_normalizations();
    size_t num_features = py::len(feature_norms);

    buffers->obs_size = num_agents * obs_height * obs_width * num_features;
    buffers->terminals_size = num_agents;
    buffers->truncations_size = num_agents;
    buffers->rewards_size = num_agents;

    // Allocate zero-initialized buffers
    buffers->observations = new ObservationType[buffers->obs_size]();
    buffers->terminals = new TerminalType[buffers->terminals_size]();
    buffers->truncations = new TruncationType[buffers->truncations_size]();
    buffers->rewards = new RewardType[buffers->rewards_size]();

    // Create numpy arrays and connect to grid
    std::vector<ssize_t> obs_shape = {static_cast<ssize_t>(num_agents),
                                      static_cast<ssize_t>(obs_height),
                                      static_cast<ssize_t>(obs_width),
                                      static_cast<ssize_t>(num_features)};
    std::vector<ssize_t> agent_shape = {static_cast<ssize_t>(num_agents)};

    auto observations = py::array_t<ObservationType, py::array::c_style>(obs_shape);
    auto terminals = py::array_t<TerminalType, py::array::c_style>(agent_shape);
    auto truncations = py::array_t<TruncationType, py::array::c_style>(agent_shape);
    auto rewards = py::array_t<RewardType, py::array::c_style>(agent_shape);

    // Copy our data into the numpy arrays
    std::copy(buffers->observations,
              buffers->observations + buffers->obs_size,
              static_cast<ObservationType*>(observations.mutable_data()));
    std::copy(buffers->terminals,
              buffers->terminals + buffers->terminals_size,
              static_cast<TerminalType*>(terminals.mutable_data()));
    std::copy(buffers->truncations,
              buffers->truncations + buffers->truncations_size,
              static_cast<TruncationType*>(truncations.mutable_data()));
    std::copy(
        buffers->rewards, buffers->rewards + buffers->rewards_size, static_cast<RewardType*>(rewards.mutable_data()));

    // Connect buffers to grid
    grid->set_buffers(observations, terminals, truncations, rewards);

    return buffers.release();
  } catch (const std::exception& e) {
    std::cerr << "Failed to allocate grid buffers: " << e.what() << std::endl;
    throw;
  }
}

// Free allocated buffers
inline void free_grid_buffers(GridBuffers* buffers) {
  if (!buffers) {
    return;
  }

  delete[] buffers->observations;
  delete[] buffers->terminals;
  delete[] buffers->truncations;
  delete[] buffers->rewards;
  delete buffers;
}

// Create a grid with buffers pre-allocated and connected
inline std::unique_ptr<MettaGrid> create_test_grid_with_buffers(uint32_t map_width = 10,
                                                                uint32_t map_height = 10,
                                                                uint32_t num_agents = 2,
                                                                uint32_t max_timestep = 100,
                                                                uint16_t obs_width = 5,
                                                                uint16_t obs_height = 5,
                                                                GridBuffers** out_buffers = nullptr) {
  // Create grid first
  auto grid = create_test_grid(map_width, map_height, num_agents, max_timestep, obs_width, obs_height);

  // Allocate and connect buffers
  GridBuffers* buffers = allocate_grid_buffers(grid.get());

  // Return buffers to caller if requested
  if (out_buffers) {
    *out_buffers = buffers;
  } else {
    // Caller doesn't want buffer management, so we don't free them
    // They'll need to be freed manually or when grid is destroyed
    // This might not be ideal - consider your use case
  }

  return grid;
}

// Create a TestMettaGrid with buffers pre-allocated and connected
inline std::unique_ptr<TestMettaGrid> create_test_mettagrid_with_buffers(uint32_t map_width = 10,
                                                                         uint32_t map_height = 10,
                                                                         uint32_t num_agents = 2,
                                                                         uint32_t max_timestep = 100,
                                                                         uint16_t obs_width = 5,
                                                                         uint16_t obs_height = 5,
                                                                         GridBuffers** out_buffers = nullptr) {
  // Create grid first
  auto grid = create_test_mettagrid(map_width, map_height, num_agents, max_timestep, obs_width, obs_height);

  // Allocate and connect buffers
  GridBuffers* buffers = allocate_grid_buffers(grid.get());

  // Return buffers to caller if requested
  if (out_buffers) {
    *out_buffers = buffers;
  }

  return grid;
}

// RAII wrapper for automatic buffer cleanup (regular MettaGrid)
class GridWithBuffers {
public:
  GridWithBuffers(uint32_t map_width = 10,
                  uint32_t map_height = 10,
                  uint32_t num_agents = 2,
                  uint32_t max_timestep = 100,
                  uint16_t obs_width = 5,
                  uint16_t obs_height = 5) {
    grid_ = create_test_grid_with_buffers(
        map_width, map_height, num_agents, max_timestep, obs_width, obs_height, &buffers_);
  }

  ~GridWithBuffers() {
    free_grid_buffers(buffers_);
  }

  // Non-copyable
  GridWithBuffers(const GridWithBuffers&) = delete;
  GridWithBuffers& operator=(const GridWithBuffers&) = delete;

  // Movable
  GridWithBuffers(GridWithBuffers&& other) noexcept : grid_(std::move(other.grid_)), buffers_(other.buffers_) {
    other.buffers_ = nullptr;
  }

  GridWithBuffers& operator=(GridWithBuffers&& other) noexcept {
    if (this != &other) {
      free_grid_buffers(buffers_);
      grid_ = std::move(other.grid_);
      buffers_ = other.buffers_;
      other.buffers_ = nullptr;
    }
    return *this;
  }

  MettaGrid* get() {
    return grid_.get();
  }
  MettaGrid* operator->() {
    return grid_.get();
  }
  MettaGrid& operator*() {
    return *grid_;
  }

  GridBuffers* buffers() {
    return buffers_;
  }

private:
  std::unique_ptr<MettaGrid> grid_;
  GridBuffers* buffers_ = nullptr;
};

// RAII wrapper for automatic buffer cleanup (TestMettaGrid)
class TestGridWithBuffers {
public:
  TestGridWithBuffers(uint32_t map_width = 10,
                      uint32_t map_height = 10,
                      uint32_t num_agents = 2,
                      uint32_t max_timestep = 100,
                      uint16_t obs_width = 5,
                      uint16_t obs_height = 5) {
    grid_ = create_test_mettagrid_with_buffers(
        map_width, map_height, num_agents, max_timestep, obs_width, obs_height, &buffers_);
  }

  ~TestGridWithBuffers() {
    free_grid_buffers(buffers_);
  }

  // Non-copyable
  TestGridWithBuffers(const TestGridWithBuffers&) = delete;
  TestGridWithBuffers& operator=(const TestGridWithBuffers&) = delete;

  // Movable
  TestGridWithBuffers(TestGridWithBuffers&& other) noexcept : grid_(std::move(other.grid_)), buffers_(other.buffers_) {
    other.buffers_ = nullptr;
  }

  TestGridWithBuffers& operator=(TestGridWithBuffers&& other) noexcept {
    if (this != &other) {
      free_grid_buffers(buffers_);
      grid_ = std::move(other.grid_);
      buffers_ = other.buffers_;
      other.buffers_ = nullptr;
    }
    return *this;
  }

  TestMettaGrid* get() {
    return grid_.get();
  }
  TestMettaGrid* operator->() {
    return grid_.get();
  }
  TestMettaGrid& operator*() {
    return *grid_;
  }

  GridBuffers* buffers() {
    return buffers_;
  }

private:
  std::unique_ptr<TestMettaGrid> grid_;
  GridBuffers* buffers_ = nullptr;
};

// Helper function to create actions as numpy array
inline py::array_t<ActionType, py::array::c_style> create_action_numpy_array(uint32_t num_agents,
                                                                             ActionType action_type = 0,
                                                                             ActionType action_arg = 0) {
  std::vector<ssize_t> shape = {static_cast<ssize_t>(num_agents), 2};
  auto actions = py::array_t<ActionType, py::array::c_style>(shape);

  auto actions_ptr = static_cast<ActionType*>(actions.mutable_data());
  for (uint32_t i = 0; i < num_agents; ++i) {
    actions_ptr[i * 2] = action_type;
    actions_ptr[i * 2 + 1] = action_arg;
  }

  return actions;
}

}  // namespace testing_utils

#endif  // METTAGRID_TESTS_TESTING_UTILS_HPP_
