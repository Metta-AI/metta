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
#include "env/mettagrid_engine.hpp"
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
    return _engine->grid();
  }
  const Actions& actions() const {
    return _actions;
  }
  const ActionSuccess& action_success() const {
    return _engine->action_success();
  }
  const ActionHandlers& action_handlers() const {
    return _engine->action_handlers();
  }

  const Agent* agent(uint32_t agent_id) const {
    return _engine->agent(agent_id);
  }

private:
  Actions _actions;

  py::array_t<uint8_t> _observations;
  py::array_t<bool> _terminals;
  py::array_t<bool> _truncations;
  py::array_t<float> _rewards;
  py::array_t<float> _episode_rewards;

  mettagrid::env::BufferSet _buffer_views;
  std::unique_ptr<mettagrid::env::MettaGridEngine> _engine;
  std::unordered_map<int, std::string> _tag_id_map;

  void refresh_buffer_views();
};

#endif  // PACKAGES_METTAGRID_CPP_BINDINGS_METTAGRID_C_HPP_
