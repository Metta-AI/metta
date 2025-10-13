#ifndef PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_
#define PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_

#define PYBIND11_DETAILED_ERROR_MESSAGES

#if defined(_WIN32)
#define METTAGRID_API __declspec(dllexport)
#else
#define METTAGRID_API __attribute__((visibility("default")))
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/mettagrid_config.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/assembler.hpp"
#include "objects/chest.hpp"
#include "systems/clipper.hpp"
#include "systems/packed_coordinate.hpp"

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
struct AssemblerConfig;
struct ChestConfig;
struct WallConfig;
struct AgentConfig;
struct GameConfig;
struct ActionConfig;
struct AttackActionConfig;
struct ChangeGlyphActionConfig;

namespace py = pybind11;

class METTAGRID_API MettaGrid {
public:
  MettaGrid(const GameConfig& cfg, py::list map, unsigned int seed);
  ~MettaGrid();

  ObservationCoord obs_width;
  ObservationCoord obs_height;

  unsigned int current_step;
  unsigned int max_steps;
  bool episode_truncates;

  std::vector<std::string> resource_names;
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
  py::dict grid_objects(int min_row = -1,
                        int max_row = -1,
                        int min_col = -1,
                        int max_col = -1,
                        const py::list& ignore_types = py::list());
  py::list action_names();

  GridCoord map_width();
  GridCoord map_height();
  py::dict feature_normalizations();
  py::dict feature_spec();
  size_t num_agents() const;
  py::none set_inventory(GridObjectId agent_id, const std::unordered_map<InventoryItem, InventoryQuantity>& inventory);
  py::array_t<float> get_episode_rewards();
  py::dict get_episode_stats();
  py::object action_space();
  py::object observation_space();
  py::list action_success_py();
  py::list max_action_args();
  py::list action_catalog();
  py::list object_type_names_py();
  py::list resource_names_py();

  uint64_t initial_grid_hash;

  using Actions = py::array_t<ActionType, py::array::c_style>;
  using ActionSuccess = std::vector<bool>;
  using ActionHandlers = std::vector<std::unique_ptr<ActionHandler>>;

  const Grid& grid() const {
    return *_grid;
  }
  const Actions& actions() const {
    return _actions;
  }
  const ActionSuccess& action_success() const {
    return _action_success;
  }
  const ActionHandlers& action_handlers() const {
    return _action_handlers;
  }

  const Agent* agent(uint32_t agent_id) const {
    return _agents[agent_id];
  }

private:
  // Member variables
  GlobalObsConfig _global_obs_config;
  GameConfig _game_config;

  std::vector<ObservationType> _resource_rewards;  // Packed inventory rewards for each agent
  std::unordered_map<unsigned int, float> _group_reward_pct;
  std::unordered_map<unsigned int, unsigned int> _group_sizes;
  std::vector<RewardType> _group_rewards;

  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;

  Actions _actions;
  ActionHandlers _action_handlers;
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

  std::unordered_map<uint8_t, float> _feature_normalizations;

  ActionSuccess _action_success;

  std::mt19937 _rng;
  unsigned int _seed;

  std::vector<std::pair<ActionType, ActionArg>> _flat_action_map;
  std::vector<std::string> _flat_action_names;
  std::vector<std::vector<int>> _action_arg_to_flat;

  // Movement tracking
  bool _track_movement_metrics;
  float _resource_loss_prob;

  // Inventory regeneration
  unsigned int _inventory_regen_interval;

  // Global systems
  std::unique_ptr<Clipper> _clipper;

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
  void build_flat_action_catalog();
  int flat_action_index(ActionType action, ActionArg arg) const;
};

#endif  // PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_
