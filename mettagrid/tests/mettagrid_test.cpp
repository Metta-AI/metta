#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>

#include "actions/attack.hpp"
#include "actions/get_output.hpp"
#include "actions/put_recipe_items.hpp"
#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"

namespace py = pybind11;

class MettaGridTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
  }

  void TearDown() override {
    // Note: We don't finalize Python here as it may be used by other tests
    // Py_Finalize() can cause issues when called multiple times
  }

  // Helper function to create test configurations for agent/object tests
  py::dict create_test_configs() {
    py::dict agent_cfg;
    agent_cfg["freeze_duration"] = 100;
    py::dict agent_rewards;
    agent_rewards["heart"] = 1.0;
    agent_rewards["ore.red"] = 0.125;  // Pick a power of 2 so floating point precision issues don't matter
    agent_cfg["rewards"] = agent_rewards;
    // higher and lower than the default
    agent_cfg["ore.red_max"] = 200;
    agent_cfg["ore.green_max"] = 100;

    py::dict group_cfg;
    group_cfg["default_item_max"] = 123;
    py::dict group_rewards;
    group_rewards["ore.red"] = 0.0;    // Should override agent ore.red reward
    group_rewards["ore.green"] = 0.5;  // New reward
    group_cfg["rewards"] = group_rewards;

    py::dict configs;
    configs["agent_cfg"] = agent_cfg;
    configs["group_cfg"] = group_cfg;
    return configs;
  }

  // Helper function to create minimal test configuration for full environment tests
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

private:
  py::scoped_interpreter guard{};  // Keep Python alive for the duration of the test
};

// ==================== Agent and Object Tests ====================

TEST_F(MettaGridTest, AgentCreation) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Test agent creation
  std::unique_ptr<Agent> agent(MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg));
  ASSERT_NE(agent, nullptr);

  // Verify merged configuration
  EXPECT_EQ(agent->freeze_duration, 100);  // Group config overrides agent

  // Verify merged rewards
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::heart], 1.0);      // Agent reward preserved
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_red], 0.0);    // Group reward overrides
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_green], 0.5);  // Group reward added
}

TEST_F(MettaGridTest, UpdateInventory) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  std::unique_ptr<Agent> agent(MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg));
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Test adding items
  int delta = agent->update_inventory(InventoryItem::heart, 5);

  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 5);

  // Test removing items
  delta = agent->update_inventory(InventoryItem::heart, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 3);

  // Test hitting zero
  delta = agent->update_inventory(InventoryItem::heart, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 0);

  // Test hitting max_items limit
  agent->update_inventory(InventoryItem::heart, 50);
  delta = agent->update_inventory(InventoryItem::heart, 200);  // max_items is 123
  EXPECT_EQ(delta, 73);                                        // Should only add up to max_items
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 123);

  delta = agent->update_inventory(InventoryItem::ore_red, 250);
  EXPECT_EQ(delta, 200);  // red has a limit of 200
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 200);

  delta = agent->update_inventory(InventoryItem::ore_green, 250);
  EXPECT_EQ(delta, 100);  // green has a limit of 100
  EXPECT_EQ(agent->inventory[InventoryItem::ore_green], 100);
}

TEST_F(MettaGridTest, AttackAction) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Create two agents - one attacker and one target
  Agent* attacker = MettaGrid::create_agent(2, 0, "red", 1, group_cfg, agent_cfg);
  Agent* target = MettaGrid::create_agent(0, 0, "blue", 2, group_cfg, agent_cfg);
  ASSERT_NE(attacker, nullptr);
  ASSERT_NE(target, nullptr);
  float attacker_reward = 0;
  attacker->init(&attacker_reward);
  float target_reward = 0;
  target->init(&target_reward);

  // Give attacker a laser
  attacker->update_inventory(InventoryItem::laser, 1);
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 1);

  // Give target some items to steal
  target->update_inventory(InventoryItem::heart, 2);
  target->update_inventory(InventoryItem::battery, 3);
  EXPECT_EQ(target->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 3);

  // Assert the attacker is facing the correct direction
  EXPECT_EQ(attacker->orientation, Orientation::Up);

  // Create a grid and add the agents to it. Then set up an attack action handler and call it.
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(attacker);
  grid.add_object(target);

  ActionConfig attack_cfg;
  Attack attack(attack_cfg);
  attack.init(&grid);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(attacker->id, 5, 0);
  EXPECT_TRUE(success);

  // Verify laser was consumed
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 0);

  // Verify target was frozen
  EXPECT_GT(target->frozen, 0);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory[InventoryItem::heart], 0);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 0);
  EXPECT_EQ(attacker->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(attacker->inventory[InventoryItem::battery], 3);

  // grid will delete the agents
}

TEST_F(MettaGridTest, PutRecipeItems) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  Agent* agent = MettaGrid::create_agent(1, 0, "red", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Create a grid and add the agent
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(agent);

  // Create a generator that takes red ore and outputs batteries
  ObjectConfig generator_cfg;
  generator_cfg["hp"] = 30;
  generator_cfg["input_ore.red"] = 1;
  generator_cfg["output_battery"] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg["max_output"] = 0;
  generator_cfg["conversion_ticks"] = 1;
  generator_cfg["cooldown"] = 10;
  generator_cfg["initial_items"] = 0;

  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(InventoryItem::ore_red, 1);
  agent->update_inventory(InventoryItem::ore_blue, 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);

  // Create put_recipe_items action handler
  ActionConfig put_cfg;
  PutRecipeItems put(put_cfg);
  put.init(&grid);

  // Test putting matching items
  bool success = put.handle_action(agent->id, 0, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 0);      // One red ore consumed
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);     // Blue ore unchanged
  EXPECT_EQ(generator->inventory[InventoryItem::ore_red], 1);  // One red ore added to generator

  // Test putting non-matching items
  success = put.handle_action(agent->id, 0, 0);
  EXPECT_FALSE(success);                                        // Should fail since we only have blue ore left
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);      // Blue ore unchanged
  EXPECT_EQ(generator->inventory[InventoryItem::ore_blue], 0);  // No blue ore in generator

  // grid will delete the agent and generator
}

TEST_F(MettaGridTest, GetOutput) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  Agent* agent = MettaGrid::create_agent(1, 0, "red", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Create a grid and add the agent
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(agent);

  // Create a generator that takes red ore and outputs batteries
  ObjectConfig generator_cfg;
  generator_cfg["hp"] = 30;
  generator_cfg["input_ore.red"] = 1;
  generator_cfg["output_battery"] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg["max_output"] = 1;
  generator_cfg["conversion_ticks"] = 1;
  generator_cfg["cooldown"] = 10;
  generator_cfg["initial_items"] = 1;

  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(InventoryItem::ore_red, 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);

  // Create get_output action handler
  ActionConfig get_cfg;
  GetOutput get(get_cfg);
  get.init(&grid);

  // Test getting output
  bool success = get.handle_action(agent->id, 0, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);      // Still have red ore
  EXPECT_EQ(agent->inventory[InventoryItem::battery], 1);      // Also have a battery
  EXPECT_EQ(generator->inventory[InventoryItem::battery], 0);  // Generator gave away its battery

  // grid will delete the agent and generator
}

// ==================== Full Environment Step Tests ====================

TEST_F(MettaGridTest, EpisodeRewards) {
  auto config = create_minimal_config();
  auto map = create_simple_map();

  // Create MettaGrid instance
  std::unique_ptr<MettaGrid> grid(new MettaGrid(config, map));

  // Create buffers for the game state
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

  // Set buffers and reset
  grid->set_buffers(observations, terminals, truncations, rewards);
  auto reset_result = grid->reset();

  // Get initial episode rewards - should be zero
  auto episode_rewards = grid->get_episode_rewards();
  auto episode_rewards_ptr = static_cast<c_rewards_type*>(episode_rewards.mutable_data());

  for (size_t i = 0; i < num_agents; i++) {
    EXPECT_FLOAT_EQ(episode_rewards_ptr[i], 0.0f) << "Episode reward should be 0 for agent " << i << " after reset";
  }

  // Create actions and take several steps
  auto actions = generate_valid_actions(grid.get(), num_agents, 0, 0);  // noop actions

  // Take first step
  auto step_result1 = grid->step(actions);

  // Get step rewards and episode rewards after first step
  auto rewards_ptr = static_cast<c_rewards_type*>(rewards.mutable_data());
  auto episode_rewards_step1 = grid->get_episode_rewards();
  auto episode_rewards_step1_ptr = static_cast<c_rewards_type*>(episode_rewards_step1.mutable_data());

  float total_step1_rewards = 0.0f;
  for (size_t i = 0; i < num_agents; i++) {
    total_step1_rewards += rewards_ptr[i];
    EXPECT_FLOAT_EQ(episode_rewards_step1_ptr[i], rewards_ptr[i])
        << "Episode reward should equal step reward for agent " << i << " after first step";
  }

  // Take second step
  auto step_result2 = grid->step(actions);

  // Get episode rewards after second step - should be cumulative
  auto episode_rewards_step2 = grid->get_episode_rewards();
  auto episode_rewards_step2_ptr = static_cast<c_rewards_type*>(episode_rewards_step2.mutable_data());

  for (size_t i = 0; i < num_agents; i++) {
    float expected_cumulative = episode_rewards_step1_ptr[i] + rewards_ptr[i];
    EXPECT_FLOAT_EQ(episode_rewards_step2_ptr[i], expected_cumulative)
        << "Episode reward should be cumulative for agent " << i << " after second step";
  }

  // Take remaining steps to verify episode rewards keep accumulating
  float previous_episode_totals[num_agents];
  for (size_t i = 0; i < num_agents; i++) {
    previous_episode_totals[i] = episode_rewards_step2_ptr[i];
  }

  // Take a few more steps
  for (int step = 2; step < 5; step++) {
    auto step_result = grid->step(actions);
    auto current_episode_rewards = grid->get_episode_rewards();
    auto current_episode_ptr = static_cast<c_rewards_type*>(current_episode_rewards.mutable_data());

    for (size_t i = 0; i < num_agents; i++) {
      float expected = previous_episode_totals[i] + rewards_ptr[i];
      EXPECT_FLOAT_EQ(current_episode_ptr[i], expected)
          << "Episode reward should continue accumulating for agent " << i << " at step " << step;
      previous_episode_totals[i] = current_episode_ptr[i];
    }
  }
}

TEST_F(MettaGridTest, BasicStepAndTerminals) {
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

  // Take a step
  auto step_result = grid->step(actions);

  // Check terminals after step - should still be 0 since we haven't hit max_steps
  for (size_t i = 0; i < num_agents; i++) {
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

  for (size_t i = 0; i < num_agents; i++) {
    EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should still be 0 for agent " << i << " at max_timestep";
    EXPECT_EQ(truncations_ptr[i], 1) << "Truncation should be 1 for agent " << i << " at max_timestep";
  }
}

TEST_F(MettaGridTest, DebugMemoryValues) {
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

  // Set some test values before calling set_buffers
  for (size_t i = 0; i < num_agents; i++) {
    terminals_ptr[i] = 99;  // Set to a distinctive value
    truncations_ptr[i] = 88;
    rewards_ptr[i] = 77.0f;
  }

  // Set buffers
  grid->set_buffers(observations, terminals, truncations, rewards);

  // Reset
  auto reset_result = grid->reset();

  // Create actions
  auto actions = generate_valid_actions(grid.get(), num_agents, 0, 0);  // force noop actions

  // Step
  auto step_result = grid->step(actions);

  // Verify the values are what we expect after the step
  for (size_t i = 0; i < num_agents; i++) {
    EXPECT_EQ(terminals_ptr[i], 0) << "Terminal should be 0 for agent " << i << " after step";
    EXPECT_EQ(truncations_ptr[i], 0) << "Truncation should be 0 for agent " << i << " after step";
    // Note: rewards might be non-zero depending on the game logic, so we don't assert specific values
  }
}