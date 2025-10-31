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
class Action;
class Agent;
class ObservationEncoder;
class GridObject;

struct GridObjectConfig;
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
  std::unordered_map<ObservationType, std::string> feature_id_to_name;

  // Python API methods
  // In general, these types need to match what puffer wants to use.
  py::tuple step();
  void set_buffers(const py::array_t<ObservationType, py::array::c_style>& observations,
                   const py::array_t<TerminalType, py::array::c_style>& terminals,
                   const py::array_t<TruncationType, py::array::c_style>& truncations,
                   const py::array_t<RewardType, py::array::c_style>& rewards,
                   const py::array_t<ActionType, py::array::c_style>& actions);
  void validate_buffers();
  py::dict grid_objects(int min_row = -1,
                        int max_row = -1,
                        int min_col = -1,
                        int max_col = -1,
                        const py::list& ignore_types = py::list());

  py::array_t<ObservationType> observations();
  py::array_t<TerminalType> terminals();
  py::array_t<TruncationType> truncations();
  py::array_t<RewardType> rewards();
  py::array_t<MaskType> masks();
  py::array_t<ActionType> actions();

  GridCoord map_width();
  GridCoord map_height();
  py::none set_inventory(GridObjectId agent_id, const std::unordered_map<InventoryItem, InventoryQuantity>& inventory);
  py::array_t<float> get_episode_rewards();
  py::dict get_episode_stats();
  py::list action_success_py();

  uint64_t initial_grid_hash;

  using Actions = py::array_t<ActionType, py::array::c_style>;
  using ActionSuccess = std::vector<bool>;

  const Grid& grid() const {
    return *_grid;
  }
  const ActionSuccess& action_success() const {
    return _action_success;
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
  std::vector<Action> _action_handlers;  // All actions from all handlers
  std::vector<std::unique_ptr<ActionHandler>> _action_handler_impl;  // Owns the ActionHandler objects
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

  // Movement tracking
  bool _track_movement_metrics;
  float _resource_loss_prob;

  // Inventory regeneration
  unsigned int _inventory_regen_interval;

  // Global systems
  std::unique_ptr<Clipper> _clipper;

  void init_action_handlers(const GameConfig& game_config);
  void add_agent(Agent* agent);
  void _init_grid(const GameConfig& game_config, const py::list& map);
  void _init_buffers(unsigned int num_agents);
  void _compute_observation(GridCoord observer_r,
                            GridCoord observer_c,
                            ObservationCoord obs_width,
                            ObservationCoord obs_height,
                            size_t agent_idx,
                            ActionType action);
  void _compute_observations(py::array_t<ActionType, py::array::c_style> actions);
  void _step();

  void _handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type);
  AgentConfig _create_agent_config(const py::dict& agent_group_cfg_py);
  WallConfig _create_wall_config(const py::dict& wall_cfg_py);
};

#endif  // PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_
