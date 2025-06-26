#ifndef METTAGRID_C_HPP_
#define METTAGRID_C_HPP_

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
#include <string>
#include <vector>

#include "types.hpp"

// Forward declarations of existing C++ classes
class Grid;
class EventManager;
class StatsTracker;
class ActionHandler;
class Agent;
class ObservationEncoder;
class GridObject;
class ConverterConfig;

namespace py = pybind11;

class METTAGRID_API MettaGrid {
public:
  MettaGrid(py::dict env_cfg, py::list map);
  ~MettaGrid();

  unsigned short obs_width;
  unsigned short obs_height;

  unsigned int current_step;
  unsigned int max_steps;

  std::vector<std::string> inventory_item_names;

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

  unsigned int map_width();
  unsigned int map_height();
  py::dict feature_normalizations();
  unsigned int num_agents();
  py::array_t<float> get_episode_rewards();
  py::dict get_episode_stats();
  py::object action_space();
  py::object observation_space();
  py::list action_success();
  py::list max_action_args();
  py::list object_type_names();
  py::list inventory_item_names_py();
  py::array_t<unsigned int> get_agent_groups() const;
  Agent* create_agent(int r, int c, const py::dict& agent_group_cfg_py);

  uint64_t initial_grid_hash;

private:
  // Member variables
  std::map<unsigned int, float> _group_reward_pct;
  std::map<unsigned int, unsigned int> _group_sizes;
  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;

  std::vector<std::unique_ptr<ActionHandler>> _action_handlers;
  int _num_action_handlers;
  std::vector<unsigned char> _max_action_args;
  unsigned char _max_action_arg;
  unsigned char _max_action_priority;

  std::unique_ptr<ObservationEncoder> _obs_encoder;
  std::unique_ptr<StatsTracker> _stats;

  unsigned int _num_observation_tokens;

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

  void init_action_handlers();
  void add_agent(Agent* agent);
  void _compute_observation(unsigned int observer_r,
                            unsigned int observer_c,
                            unsigned short obs_width,
                            unsigned short obs_height,
                            size_t agent_idx,
                            ActionType action,
                            ActionArg action_arg);
  void _compute_observations(py::array_t<ActionType, py::array::c_style> actions);
  void _step(py::array_t<ActionType, py::array::c_style> actions);

  void _handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type, ActionArg arg);
  ConverterConfig _create_converter_config(const py::dict& converter_cfg_py);
};

#endif  // METTAGRID_C_HPP_
