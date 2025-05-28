#ifndef METTAGRID_METTAGRID_METTAGRID_C_HPP_
#define METTAGRID_METTAGRID_METTAGRID_C_HPP_

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

namespace py = pybind11;

class METTAGRID_API MettaGrid {
public:
  MettaGrid(py::dict env_cfg, py::list map);
  ~MettaGrid();

  unsigned short obs_width;
  unsigned short obs_height;

  unsigned int current_step;
  unsigned int max_steps;

  // Python API methods
  py::tuple reset();
  py::tuple step(py::array_t<int> actions);
  void set_buffers(py::array_t<c_observations_type, py::array::c_style>& observations,
                   py::array_t<c_terminals_type, py::array::c_style>& terminals,
                   py::array_t<c_truncations_type, py::array::c_style>& truncations,
                   py::array_t<c_rewards_type, py::array::c_style>& rewards);

  void validate_buffers();
  py::dict grid_objects();
  py::list action_names();

  unsigned int map_width() const;
  unsigned int map_height() const;
  py::list grid_features();
  unsigned int num_agents() const;
  py::array_t<c_rewards_type> get_episode_rewards();
  py::dict get_episode_stats();
  py::object action_space();
  py::object observation_space();
  py::list action_success();
  py::list max_action_args();
  py::list object_type_names();
  py::list inventory_item_names();
  static Agent* create_agent(int r,
                             int c,
                             const std::string& group_name,
                             unsigned int group_id,
                             const py::dict& group_cfg_py,
                             const py::dict& agent_cfg_py);

  static std::string cpp_get_numpy_type_name(const char* type_id);

  bool is_gym_mode() const {
    return _gym_mode;
  }

private:
  // Member variables
  py::dict _cfg;
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

  bool _use_observation_tokens;
  unsigned int _num_observation_tokens;  // Added missing member variable

  // TODO: currently these are owned and destroyed by the grid, but we should
  // probably move ownership here.
  std::vector<Agent*> _agents;

  // Mode flags
  bool _gym_mode;
  bool _set_buffers_called;

  // Internal buffers for gym mode
  std::unique_ptr<c_observations_type[]> _internal_observations;
  std::unique_ptr<c_terminals_type[]> _internal_terminals;
  std::unique_ptr<c_truncations_type[]> _internal_truncations;
  std::unique_ptr<c_rewards_type[]> _internal_rewards;

  // Pointers to external buffers - these are required and must be set
  c_observations_type* _observations;
  c_terminals_type* _terminals;
  c_truncations_type* _truncations;
  c_rewards_type* _rewards;

  // Buffer sizes
  size_t _observations_size;
  size_t _terminals_size;
  size_t _truncations_size;
  size_t _rewards_size;

  // Internal buffers
  std::vector<float> _episode_rewards;

  std::vector<std::string> _grid_features;

  std::vector<bool> _action_success;

  void init_action_handlers();
  void add_agent(Agent* agent);
  void _compute_observation(unsigned int observer_r,
                            unsigned int observer_c,
                            unsigned short obs_width,
                            unsigned short obs_height,
                            size_t agent_idx);
  void _compute_observations(py::array_t<int> actions);
  void _step(py::array_t<int> actions);

  void _allocate_internal_buffers();
  void _free_internal_buffers();
  void _setup_gym_mode();
};

#endif  // METTAGRID_METTAGRID_METTAGRID_C_HPP_
