#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

namespace py = pybind11;

class MettaGridStepTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Python interpreter
    Py_Initialize();
  }

  void TearDown() override {
    // Finalize Python interpreter
    Py_Finalize();
  }

  // Helper function to create minimal test configuration
  py::dict create_minimal_config() {
    py::dict env_cfg;

    // Game config
    py::dict game_cfg;
    game_cfg["num_agents"] = 2;
    game_cfg["max_steps"] = 10;
    game_cfg["obs_width"] = 5;
    game_cfg["obs_height"] = 5;

    // Actions config - this needs to be under game, not top level
    py::dict actions;
    py::dict noop_action;
    noop_action["enabled"] = true;
    actions["noop"] = noop_action;

    py::dict move_action;
    move_action["enabled"] = true;
    actions["move"] = move_action;

    py::dict put_items;
    put_items["enabled"] = false;
    actions["put_items"] = put_items;

    py::dict get_items;
    get_items["enabled"] = false;
    actions["get_items"] = get_items;

    py::dict rotate_action;
    rotate_action["enabled"] = false;
    actions["rotate"] = rotate_action;

    py::dict attack_action;
    attack_action["enabled"] = false;
    actions["attack"] = attack_action;

    py::dict swap_action;
    swap_action["enabled"] = false;
    actions["swap"] = swap_action;

    py::dict change_color_action;
    change_color_action["enabled"] = false;
    actions["change_color"] = change_color_action;

    game_cfg["actions"] = actions;

    // Groups config
    py::dict groups;
    py::dict group1;
    group1["id"] = 1;
    group1["group_reward_pct"] = 0.1;
    py::dict group1_props;
    group1_props["max_inventory"] = 50;
    group1["props"] = group1_props;
    groups["red"] = group1;

    py::dict group2;
    group2["id"] = 2;
    group2["group_reward_pct"] = 0.0;
    py::dict group2_props;
    group2_props["max_inventory"] = 50;
    group2["props"] = group2_props;
    groups["blue"] = group2;
    game_cfg["groups"] = groups;

    // Agent config
    py::dict agent_cfg;
    agent_cfg["freeze_duration"] = 100;
    agent_cfg["max_inventory"] = 50;
    py::dict agent_rewards;
    agent_rewards["heart"] = 1.0;
    agent_cfg["rewards"] = agent_rewards;
    game_cfg["agent"] = agent_cfg;

    // Objects config
    py::dict objects;
    py::dict wall_cfg;
    wall_cfg["hp"] = 100;
    objects["wall"] = wall_cfg;

    py::dict block_cfg;
    block_cfg["hp"] = 50;
    objects["block"] = block_cfg;
    game_cfg["objects"] = objects;

    env_cfg["game"] = game_cfg;
    return env_cfg;
  }

  // Helper function to create a simple 5x5 map with 2 agents
  py::list create_simple_map() {
    py::list map;

    // Create a 5x5 grid
    for (int r = 0; r < 5; r++) {
      py::list row;
      for (int c = 0; c < 5; c++) {
        if (r == 1 && c == 1) {
          row.append("agent.red");
        } else if (r == 3 && c == 3) {
          row.append("agent.blue");
        } else if (r == 0 || r == 4 || c == 0 || c == 4) {
          row.append("wall");
        } else {
          row.append("empty");
        }
      }
      map.append(row);
    }
    return map;
  }

  // C++ equivalent of generate_valid_random_actions
  py::array_t<int> generate_valid_actions(MettaGrid* grid,
                                          size_t num_agents,
                                          int force_action_type = -1,
                                          int force_action_arg = -1,
                                          int seed = -1) {
    if (seed >= 0) {
      srand(seed);
    }

    // Get action space info
    auto action_names = grid->action_names();
    size_t num_actions = action_names.size();
    auto max_args = grid->max_action_args();

    // Create actions array
    std::vector<ssize_t> action_shape = {static_cast<ssize_t>(num_agents), 2};
    auto actions = py::array_t<int>(action_shape);
    auto actions_ptr = static_cast<int*>(actions.mutable_data());

    for (size_t i = 0; i < num_agents; i++) {
      int act_type, act_arg;

      // Determine action type
      if (force_action_type >= 0) {
        act_type = std::min(force_action_type, static_cast<int>(num_actions - 1));
      } else {
        act_type = rand() % num_actions;
      }

      // Get maximum allowed argument for this action type
      int max_allowed = 0;
      if (act_type < static_cast<int>(max_args.size())) {
        max_allowed = max_args[act_type].cast<int>();
      }

      // Determine action argument
      if (force_action_arg >= 0) {
        act_arg = std::min(force_action_arg, max_allowed);
      } else {
        act_arg = max_allowed > 0 ? rand() % (max_allowed + 1) : 0;
      }

      // Set the action values
      actions_ptr[i * 2 + 0] = act_type;
      actions_ptr[i * 2 + 1] = act_arg;
    }

    return actions;
  }
};

TEST_F(MettaGridStepTest, BasicStepAndTerminals) {
  auto config = create_minimal_config();
  auto map = create_simple_map();

  // Create MettaGrid instance
  std::unique_ptr<MettaGrid> grid(new MettaGrid(config, map));

  // Create buffers for the game state
  size_t num_agents = grid->num_agents();
  size_t obs_features = grid->grid_features().size();

  // Allocate numpy arrays for game state
  std::vector<ssize_t> obs_shape = {static_cast<ssize_t>(num_agents),
                                    static_cast<ssize_t>(grid->obs_height),
                                    static_cast<ssize_t>(grid->obs_width),
                                    static_cast<ssize_t>(obs_features)};
  std::vector<ssize_t> scalar_shape = {static_cast<ssize_t>(num_agents)};

  auto observations = py::array_t<c_observations_type, py::array::c_style>(obs_shape);
  auto terminals = py::array_t<c_terminals_type, py::array::c_style>(scalar_shape);
  auto truncations = py::array_t<c_truncations_type, py::array::c_style>(scalar_shape);
  auto rewards = py::array_t<c_rewards_type, py::array::c_style>(scalar_shape);

  // Set buffers
  grid->set_buffers(observations, terminals, truncations, rewards);

  // Reset the environment
  auto reset_result = grid->reset();

  // Verify initial state - terminals should be 0
  auto terminals_ptr = static_cast<c_terminals_type*>(terminals.mutable_data());
  for (size_t i = 0; i < num_agents; i++) {
    EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should be 0 for agent " << i << " after reset";
  }

  // Create action array - both agents do noop (action 0, arg 0)
  auto actions = generate_valid_actions(grid.get(), num_agents, 0, 0);  // force noop actions

  // Debug: print action names and generated actions
  auto action_names = grid->action_names();
  std::cout << "Available actions: ";
  for (size_t i = 0; i < action_names.size(); i++) {
    std::cout << i << "=" << action_names[i].cast<std::string>() << " ";
  }
  std::cout << std::endl;

  auto actions_ptr = static_cast<int*>(actions.mutable_data());
  std::cout << "Generated actions: ";
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "agent" << i << "=(" << actions_ptr[i * 2] << "," << actions_ptr[i * 2 + 1] << ") ";
  }
  std::cout << std::endl;

  // Take a step
  auto step_result = grid->step(actions);

  // Check terminals after step - should still be 0 since we haven't hit max_steps
  std::cout << "Current timestep: " << grid->current_timestep << std::endl;
  std::cout << "Max timestep: " << grid->max_timestep << std::endl;

  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "Agent " << i << " terminal: " << static_cast<int>(terminals_ptr[i]) << std::endl;
    EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should be 0 for agent " << i << " after one step";
  }

  // Take steps until we hit max_steps
  for (unsigned int step = 1; step < grid->max_timestep; step++) {
    auto step_result = grid->step(actions);

    // Terminals should still be 0 until max_timestep
    for (size_t i = 0; i < num_agents; i++) {
      EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should be 0 for agent " << i << " at step " << step;
    }
  }

  // Take one more step to hit max_timestep
  auto final_step = grid->step(actions);

  // Now check truncations (should be 1) and terminals (should still be 0)
  auto truncations_ptr = static_cast<c_truncations_type*>(truncations.mutable_data());

  std::cout << "Final timestep: " << grid->current_timestep << std::endl;
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "Agent " << i << " terminal: " << static_cast<int>(terminals_ptr[i])
              << ", truncation: " << static_cast<int>(truncations_ptr[i]) << std::endl;

    EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should still be 0 for agent " << i << " at max_timestep";
    EXPECT_EQ(truncations_ptr[i], 1) << "Truncation should be 1 for agent " << i << " at max_timestep";
  }
}

TEST_F(MettaGridStepTest, DebugMemoryValues) {
  auto config = create_minimal_config();
  auto map = create_simple_map();

  // Create MettaGrid instance
  std::unique_ptr<MettaGrid> grid(new MettaGrid(config, map));

  // Create buffers
  size_t num_agents = grid->num_agents();
  size_t obs_features = grid->grid_features().size();

  std::vector<ssize_t> obs_shape = {static_cast<ssize_t>(num_agents),
                                    static_cast<ssize_t>(grid->obs_height),
                                    static_cast<ssize_t>(grid->obs_width),
                                    static_cast<ssize_t>(obs_features)};
  std::vector<ssize_t> scalar_shape = {static_cast<ssize_t>(num_agents)};

  auto observations = py::array_t<c_observations_type, py::array::c_style>(obs_shape);
  auto terminals = py::array_t<c_terminals_type, py::array::c_style>(scalar_shape);
  auto truncations = py::array_t<c_truncations_type, py::array::c_style>(scalar_shape);
  auto rewards = py::array_t<c_rewards_type, py::array::c_style>(scalar_shape);

  // Get raw pointers for debugging
  auto terminals_ptr = static_cast<c_terminals_type*>(terminals.mutable_data());
  auto truncations_ptr = static_cast<c_truncations_type*>(truncations.mutable_data());
  auto rewards_ptr = static_cast<c_rewards_type*>(rewards.mutable_data());

  std::cout << "Buffer addresses:" << std::endl;
  std::cout << "  terminals: " << static_cast<void*>(terminals_ptr) << std::endl;
  std::cout << "  truncations: " << static_cast<void*>(truncations_ptr) << std::endl;
  std::cout << "  rewards: " << static_cast<void*>(rewards_ptr) << std::endl;

  // Set some test values before calling set_buffers
  for (size_t i = 0; i < num_agents; i++) {
    terminals_ptr[i] = 99;  // Set to a distinctive value
    truncations_ptr[i] = 88;
    rewards_ptr[i] = 77.0f;
  }

  std::cout << "Before set_buffers:" << std::endl;
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "  Agent " << i << ": terminal=" << static_cast<int>(terminals_ptr[i])
              << ", truncation=" << static_cast<int>(truncations_ptr[i]) << ", reward=" << rewards_ptr[i] << std::endl;
  }

  // Set buffers
  grid->set_buffers(observations, terminals, truncations, rewards);

  std::cout << "After set_buffers:" << std::endl;
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "  Agent " << i << ": terminal=" << static_cast<int>(terminals_ptr[i])
              << ", truncation=" << static_cast<int>(truncations_ptr[i]) << ", reward=" << rewards_ptr[i] << std::endl;
  }

  // Reset
  auto reset_result = grid->reset();

  std::cout << "After reset:" << std::endl;
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "  Agent " << i << ": terminal=" << static_cast<int>(terminals_ptr[i])
              << ", truncation=" << static_cast<int>(truncations_ptr[i]) << ", reward=" << rewards_ptr[i] << std::endl;
  }

  // Create actions
  auto actions = generate_valid_actions(grid.get(), num_agents, 0, 0);  // force noop actions

  // Debug: show what actions we're using
  auto actions_ptr = static_cast<int*>(actions.mutable_data());
  std::cout << "Using actions: ";
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "agent" << i << "=(" << actions_ptr[i * 2] << "," << actions_ptr[i * 2 + 1] << ") ";
  }
  std::cout << std::endl;

  // Step
  auto step_result = grid->step(actions);

  std::cout << "After step:" << std::endl;
  for (size_t i = 0; i < num_agents; i++) {
    std::cout << "  Agent " << i << ": terminal=" << static_cast<int>(terminals_ptr[i])
              << ", truncation=" << static_cast<int>(truncations_ptr[i]) << ", reward=" << rewards_ptr[i] << std::endl;
  }
}