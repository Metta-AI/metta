#ifndef METTAGRID_C_HPP_
#define METTAGRID_C_HPP_

#define PYBIND11_DETAILED_ERROR_MESSAGES

#if defined(_WIN32)
#define METTAGRID_API __declspec(dllexport)
#else
#define METTAGRID_API __attribute__((visibility("default")))
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "grid_object.hpp"
#include "packed_coordinate.hpp"
#include "types.hpp"

// Forward declarations of existing C++ classes
class Grid;
class EventManager;
class StatsTracker;
class ActionHandler;
class Agent;
class ObservationEncoder;
class GridObject;

struct GridObjectConfig;
struct ConverterConfig;
struct WallConfig;
struct AgentConfig;
struct GameConfig;
struct ActionConfig;
struct AttackActionConfig;
struct ChangeGlyphActionConfig;

namespace py = pybind11;

struct GlobalObsConfig {
  bool episode_completion_pct = true;
  bool last_action = true;  // Controls both last_action and last_action_arg
  bool last_reward = true;
  bool resource_rewards = false;  // Controls whether resource rewards are included in observations
};

struct GameConfig {
  size_t num_agents;
  unsigned int max_steps;
  bool episode_truncates;
  ObservationCoord obs_width;
  ObservationCoord obs_height;
  std::vector<std::string> inventory_item_names;
  unsigned int num_observation_tokens;
  GlobalObsConfig global_obs;
  std::map<std::string, std::shared_ptr<ActionConfig>> actions;
  std::map<std::string, std::shared_ptr<GridObjectConfig>> objects;
  bool show_recipe_inputs = false;
};

class METTAGRID_API MettaGrid {
public:
  MettaGrid(const GameConfig& cfg, py::list map, unsigned int seed);
  ~MettaGrid();

  ObservationCoord obs_width;
  ObservationCoord obs_height;

  unsigned int current_step;
  unsigned int max_steps;
  bool episode_truncates;

  std::vector<std::string> inventory_item_names;
  std::vector<std::string> object_type_names;

  // Python API methods
  py::tuple reset();
  // In general, these types need to match what puffer wants to use.
  py::tuple step(py::array_t<ActionType, py::array::c_style> actions);
  void set_buffers(const py::array_t<ObservationType, py::array::c_style>& observations,
                   const py::array_t<TerminalType, py::array::c_style>& terminals,
                   const py::array_t<TruncationType, py::array::c_style>& truncations,
                   const py::array_t<RewardType, py::array::c_style>& rewards);
  void validate_buffers();
  py::dict grid_objects();
  py::list action_names();

  GridCoord map_width();
  GridCoord map_height();
  py::dict feature_normalizations();
  py::dict feature_spec();
  size_t num_agents();
  py::array_t<float> get_episode_rewards();
  py::dict get_episode_stats();
  py::object action_space();
  py::object observation_space();
  py::list action_success();
  py::list max_action_args();
  py::list object_type_names_py();
  py::list inventory_item_names_py();
  py::array_t<unsigned int> get_agent_groups() const;

  uint64_t initial_grid_hash;

private:
  // Member variables
  GlobalObsConfig _global_obs_config;
  std::vector<ObservationType> _resource_rewards;  // Packed inventory rewards for each agent
  std::map<unsigned int, float> _group_reward_pct;
  std::map<unsigned int, unsigned int> _group_sizes;
  std::vector<RewardType> _group_rewards;

  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;

  std::vector<std::unique_ptr<ActionHandler>> _action_handlers;
  size_t _num_action_handlers;
  std::vector<unsigned char> _max_action_args;
  unsigned char _max_action_arg;
  unsigned char _max_action_priority;

  std::unique_ptr<ObservationEncoder> _obs_encoder;
  std::unique_ptr<StatsTracker> _stats;

  size_t _num_observation_tokens;

  // TODO: currently these are owned and destroyed by the grid, but we should
  // probably move ownership here.
  std::vector<Agent*> _agents;

  // We'd prefer to store these as more raw c-style arrays, but we need to both
  // operate on the memory directly and return them to python.
  py::array_t<uint8_t> _observations;
  py::array_t<bool> _terminals;
  py::array_t<bool> _truncations;
  py::array_t<float> _rewards;
  py::array_t<float> _episode_rewards;

  std::map<uint8_t, float> _feature_normalizations;

  std::vector<bool> _action_success;

  std::mt19937 _rng;
  unsigned int _seed;

  void init_action_handlers();
  void add_agent(Agent* agent);
  void _compute_observation(GridCoord observer_r,
                            GridCoord observer_c,
                            ObservationCoord obs_width,
                            ObservationCoord obs_height,
                            size_t agent_idx,
                            ActionType action,
                            ActionArg action_arg);
  void _compute_observations(py::array_t<ActionType, py::array::c_style> actions);
  void _step(py::array_t<ActionType, py::array::c_style> actions);

  void _handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type, ActionArg arg);
  AgentConfig _create_agent_config(const py::dict& agent_group_cfg_py);
  ConverterConfig _create_converter_config(const py::dict& converter_cfg_py);
  WallConfig _create_wall_config(const py::dict& wall_cfg_py);
};

#endif  // METTAGRID_C_HPP_
