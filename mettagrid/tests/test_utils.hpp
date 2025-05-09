#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.hpp"
#include "objects/agent.hpp"

namespace test_utils {

// Create a simple test map with two agents (red and blue) at opposite corners
inline std::string create_test_map_json() {
  std::string map_json = R"([
        ["agent.red", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "agent.blue", "empty"]
    ])";
  return map_json;
}

// Create a basic config with two agent groups and standard objects
inline std::string create_test_config_json() {
  std::string config_json = R"({
        "agent": {
            "health": 100,
            "max_health": 100,
            "energy": 100,
            "max_energy": 100
        },
        "groups": {
            "red": {
                "id": 0,
                "group_reward_pct": 0.2,
                "props": {
                    "color": 1
                }
            },
            "blue": {
                "id": 1,
                "group_reward_pct": 0.2,
                "props": {
                    "color": 2
                }
            }
        },
        "objects": {
            "wall": { "health": 100 },
            "block": { "health": 50 },
            "mine.red": { "health": 80 },
            "generator.red": { "health": 80 },
            "altar": { "health": 100 },
            "armory": { "health": 100 },
            "lasery": { "health": 100 },
            "lab": { "health": 100 },
            "factory": { "health": 100 },
            "temple": { "health": 100 }
        },
        "actions": {
            "noop": {
                "enabled": true,
                "priority": 0
            },
            "move": {
                "enabled": true,
                "priority": 1
            },
            "rotate": {
                "enabled": true,
                "priority": 1
            },
            "attack": {
                "enabled": true,
                "priority": 2
            },
            "put_items": {
                "enabled": true,
                "priority": 2
            },
            "get_items": {
                "enabled": true,
                "priority": 2
            },
            "swap": {
                "enabled": true,
                "priority": 1
            },
            "change_color": {
                "enabled": true, 
                "priority": 1
            }
        }
    })";
  return config_json;
}

// Create a standard action array for testing
inline int32_t** create_action_array(uint32_t num_agents, uint32_t action_type = 0, uint32_t action_arg = 0) {
  int32_t** actions = new int32_t*[num_agents];
  for (uint32_t i = 0; i < num_agents; ++i) {
    actions[i] = new int32_t[2];
    actions[i][0] = action_type;
    actions[i][1] = action_arg;
  }
  return actions;
}

// Clean up action array
inline void delete_action_array(int32_t** actions, uint32_t num_agents) {
  for (uint32_t i = 0; i < num_agents; ++i) {
    delete[] actions[i];
  }
  delete[] actions;
}

// Create a standard MettaGrid for testing with default dimensions
inline std::unique_ptr<CppMettaGrid> create_test_grid(uint32_t map_width = 10,
                                                      uint32_t map_height = 10,
                                                      uint32_t num_agents = 2,
                                                      uint32_t max_timestep = 100,
                                                      uint16_t obs_width = 5,
                                                      uint16_t obs_height = 5) {
  // Create grid with specified dimensions
  auto grid = std::make_unique<CppMettaGrid>(map_width, map_height, num_agents, max_timestep, obs_width, obs_height);

  // Initialize from standard test map and config
  grid->initialize_from_json(create_test_map_json(), create_test_config_json());

  return grid;
}

// Create a MettaGrid from saved test data files
inline std::unique_ptr<CppMettaGrid> create_grid_from_test_files(
    const std::string& map_path = "test_data/mettagrid_test_args_env_map.txt",
    const std::string& config_path = "test_data/mettagrid_test_args_env_cfg.json",
    uint32_t max_timestep = 1000,
    uint16_t obs_width = 11,
    uint16_t obs_height = 11) {
  // Read the map file
  std::ifstream map_file(map_path);
  if (!map_file.is_open()) {
    throw std::runtime_error("Could not open map file: " + map_path);
  }

  // Parse map dimensions and content
  std::vector<std::vector<std::string>> map_data;
  std::string line;
  while (std::getline(map_file, line)) {
    std::vector<std::string> row;
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      row.push_back(cell);
    }
    if (!row.empty()) {
      map_data.push_back(row);
    }
  }

  // Determine map dimensions
  uint32_t map_height = map_data.size();
  uint32_t map_width = map_data.empty() ? 0 : map_data[0].size();

  // Count agents in the map
  uint32_t num_agents = 0;
  for (const auto& row : map_data) {
    for (const auto& cell : row) {
      // Count cells that contain "agent" as agents
      if (cell.find("agent") != std::string::npos) {
        num_agents++;
      }
    }
  }

  // Read the config file
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    throw std::runtime_error("Could not open config file: " + config_path);
  }

  // Read the entire file into a string
  std::string config_str((std::istreambuf_iterator<char>(config_file)), std::istreambuf_iterator<char>());

  // Parse the config JSON
  nlohmann::json config = nlohmann::json::parse(config_str);

  // Create the map JSON
  nlohmann::json map_json = nlohmann::json::array();
  for (const auto& row : map_data) {
    map_json.push_back(row);
  }

  // Create the grid with dimensions from the map
  auto grid = std::make_unique<CppMettaGrid>(map_width, map_height, num_agents, max_timestep, obs_width, obs_height);

  // Initialize from the loaded map and config
  grid->initialize_from_json(map_json.dump(), config["game"].dump());

  return grid;
}

// Create a MettaGrid from the mettagrid_test_args specifically
inline std::unique_ptr<CppMettaGrid> create_grid_from_mettagrid_args(const std::string& test_data_dir = "./test_data") {
  std::string map_path = test_data_dir + "/mettagrid_test_args_env_map.txt";
  std::string config_path = test_data_dir + "/mettagrid_test_args_env_cfg.json";

  return create_grid_from_test_files(map_path, config_path);
}

// Structure to hold all buffer pointers for easier management
struct GridBuffers {
  ObsType* observations = nullptr;
  int8_t* terminals = nullptr;
  int8_t* truncations = nullptr;
  float* rewards = nullptr;

  // Size information for reference
  size_t obs_size = 0;
  size_t terminals_size = 0;
  size_t truncations_size = 0;
  size_t rewards_size = 0;
};

// Helper function to allocate buffers for a grid
inline GridBuffers* allocate_grid_buffers(CppMettaGrid* grid) {
  if (!grid) {
    throw std::invalid_argument("Cannot allocate buffers for null grid");
  }

  GridBuffers* buffers = new GridBuffers();

  // Get exact buffer sizes from the grid
  buffers->obs_size = grid->get_observations_size();
  buffers->terminals_size = grid->get_terminals_size();
  buffers->truncations_size = grid->get_truncations_size();
  buffers->rewards_size = grid->get_rewards_size();

  // Allocate memory for each buffer with zero initialization
  buffers->observations = new ObsType[buffers->obs_size]();
  buffers->terminals = new int8_t[buffers->terminals_size]();
  buffers->truncations = new int8_t[buffers->truncations_size]();
  buffers->rewards = new float[buffers->rewards_size]();

  // Set the buffers in the grid
  grid->set_buffers(buffers->observations, buffers->terminals, buffers->truncations, buffers->rewards);

  return buffers;
}

// Helper function to free allocated buffers
inline void free_grid_buffers(GridBuffers* buffers) {
  if (!buffers) {
    return;  // Nothing to free
  }

  // Free all buffer memory
  delete[] buffers->observations;
  delete[] buffers->terminals;
  delete[] buffers->truncations;
  delete[] buffers->rewards;

  // Delete the buffers struct itself
  delete buffers;
}

// Helper function to create a grid with buffers already set up
inline std::unique_ptr<CppMettaGrid> create_test_grid_with_buffers(uint32_t map_width = 10,
                                                                   uint32_t map_height = 10,
                                                                   uint32_t num_agents = 2,
                                                                   uint32_t max_timestep = 100,
                                                                   uint16_t obs_width = 5,
                                                                   uint16_t obs_height = 5,
                                                                   GridBuffers** out_buffers = nullptr) {
  // Create the grid
  auto grid = create_test_grid(map_width, map_height, num_agents, max_timestep, obs_width, obs_height);

  // Allocate and set up buffers
  GridBuffers* buffers = allocate_grid_buffers(grid.get());

  // Return the buffers to the caller if requested
  if (out_buffers) {
    *out_buffers = buffers;
  } else {
    // If caller doesn't want the buffers struct, we need to free them
    free_grid_buffers(buffers);
  }

  return grid;
}

}  // namespace test_utils

#endif  // TEST_UTILS_HPP