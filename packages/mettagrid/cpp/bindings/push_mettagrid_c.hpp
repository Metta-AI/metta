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

#include <array>
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
struct ChangeVibeActionConfig;

struct LocationSpan {
  size_t start;
  size_t len;
};

namespace py = pybind11;

class METTAGRID_API MettaGrid {
public:
  MettaGrid(const GameConfig& cfg, py::list map, unsigned int seed);
  ~MettaGrid();

  enum DirtyBits : uint8_t {
    kDirtyLocation = 1 << 0,
    kDirtyContent = 1 << 1,
    kDirtyAll = kDirtyLocation | kDirtyContent,
  };

  static constexpr size_t kMaxTokensPerCell = 24;

  struct CellCache {
    std::array<ObservationType, kMaxTokensPerCell> feature_ids{};
    std::array<ObservationType, kMaxTokensPerCell> values{};
    uint8_t static_count = 0;
    uint8_t dynamic_count = 0;
  };

  struct PackedOffset {
    int16_t dr;
    int16_t dc;
    uint8_t packed;
  };

  using Actions = py::array_t<ActionType, py::array::c_style>;
  using ActionSuccess = std::vector<bool>;

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
  void step();
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

  enum class ActionDirtyKind : uint8_t { kMove, kChangeVibe, kNoop, kOther };

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
  GlobalObsConfig _global_obs_config;
  GameConfig _game_config;
  size_t _num_observation_tokens;
  unsigned int _inventory_regen_interval;
  unsigned int _seed;
  std::mt19937 _rng;

  std::unique_ptr<Grid> _grid;
  std::unique_ptr<ObservationEncoder> _obs_encoder;
  std::unique_ptr<StatsTracker> _stats;
  std::unique_ptr<Clipper> _clipper;

  // TODO: currently these are owned and destroyed by the grid, but we should
  // probably move ownership here.
  std::vector<Agent*> _agents;

  Actions _actions;
  std::vector<Action> _action_handlers;                              // All actions from all handlers
  std::vector<std::unique_ptr<ActionHandler>> _action_handler_impl;  // Owns the ActionHandler objects
  std::vector<ActionDirtyKind> _action_dirty_kinds;
  unsigned char _max_action_priority;
  ActionSuccess _action_success;
  std::vector<size_t> _agent_indices;
  std::vector<ActionType> _executed_actions;
  std::vector<size_t> _assembler_cells;

  std::vector<CellCache> _cell_cache;  // size: grid_height * grid_width
  std::vector<uint8_t> _dirty_flags;
  std::vector<size_t> _dirty_cells;
  std::vector<PackedOffset> _obs_pattern;
  bool _logged_cell_truncation = false;
  std::unordered_map<std::string, size_t> _resource_name_to_index;
  std::vector<uint8_t> _goal_token_flags;
  std::vector<PartialObservationToken> _global_tokens_buffer;

  py::array_t<uint8_t> _observations;
  py::array_t<bool> _terminals;
  py::array_t<bool> _truncations;
  py::array_t<float> _rewards;
  py::array_t<float> _episode_rewards;

  void init_action_handlers(const GameConfig& game_config);
  void add_agent(Agent* agent);
  void _init_grid(const GameConfig& game_config, const py::list& map);
  void _make_buffers(unsigned int num_agents);
  void _init_buffers(unsigned int num_agents);
  void _compute_observation(GridCoord observer_r,
                            GridCoord observer_c,
                            ObservationCoord obs_width,
                            ObservationCoord obs_height,
                            size_t agent_idx,
                            ActionType action);
  void _compute_observations(const std::vector<ActionType>& executed_actions);
  void _step();

  void _handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type);
  AgentConfig _create_agent_config(const py::dict& agent_group_cfg_py);
  WallConfig _create_wall_config(const py::dict& wall_cfg_py);

  inline size_t _cell_index(GridCoord r, GridCoord c) const { return static_cast<size_t>(r) * _grid->width + c; }
  inline void _mark_cell_dirty(GridCoord r, GridCoord c, uint8_t flags = DirtyBits::kDirtyAll);
  inline void _mark_observation_window_dirty(GridCoord center_r, GridCoord center_c, uint8_t flags = DirtyBits::kDirtyAll);
  void _refresh_dirty_cells();
  void _mark_all_assembler_cells_dirty(bool force = false);
  inline void _mark_if_assembler(GridCoord r, GridCoord c);
  inline void _mark_adjacent_assemblers(GridCoord r, GridCoord c);

  bool _is_global_feature(ObservationType feature_id) const;

  // Push-mode helpers and state
  std::vector<std::vector<size_t>> _cell_to_agents;
  void _rebuild_fov_reverse_map();

  std::vector<std::array<LocationSpan, 256>> _location_spans;
  void _rebuild_location_spans();
  void _rebuild_location_spans_for_agent(size_t agent_idx);

  void _init_cell_cache();
  void _refresh_cell_cache(GridCoord r, GridCoord c);

  void _update_observation_from_cache(GridCoord r, GridCoord c);
  void _rewrite_global_tokens(size_t agent_idx, ActionType action);
  void _clear_agent_observation(size_t agent_idx);

  std::vector<GridLocation> _prev_locations;
};

#endif  // PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_
