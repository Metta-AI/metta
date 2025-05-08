#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

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

}  // namespace test_utils

#endif  // TEST_UTILS_HPP