#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_METTAGRID_ENGINE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_METTAGRID_ENGINE_HPP_

#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/mettagrid_config_types.hpp"
#include "core/types.hpp"
#include "env/buffer_views.hpp"
#include "objects/agent.hpp"

class Grid;
class EventManager;
class StatsTracker;
class ActionHandler;
class ObservationEncoder;
class Clipper;

namespace mettagrid::env {

// Lightweight view over a 2-column action matrix (action type, argument).
struct ActionMatrixView {
  const ActionType* data;
  size_t num_agents;

  const ActionType* row(size_t idx) const {
    return data + idx * 2;
  }
};

// Engine that mirrors the gameplay loop implemented in the pybind MettaGrid class, but without
// any dependency on pybind11 containers. This will back both the socket server and the Python bindings.
class MettaGridEngine {
public:
  MettaGridEngine(const GameConfig& config,
                  const std::vector<std::vector<std::string>>& map,
                  unsigned int seed);
  ~MettaGridEngine();

  MettaGridEngine(const MettaGridEngine&) = delete;
  MettaGridEngine& operator=(const MettaGridEngine&) = delete;

  void set_buffers(const BufferSet& buffers);
  void validate_buffers() const;

  void reset();
  void step(ActionMatrixView actions);

  ObservationCoord observation_width() const {
    return obs_width;
  }

  ObservationCoord observation_height() const {
    return obs_height;
  }

  size_t num_agents() const {
    return _agents.size();
  }

  const BufferSet& buffers() const {
    return _buffers;
  }

  const std::vector<bool>& action_success() const {
    return _action_success;
  }

  const Grid& grid() const;
  Grid& mutable_grid();
  StatsTracker& stats();
  const StatsTracker& stats() const;
  const std::vector<std::unique_ptr<ActionHandler>>& action_handlers() const {
    return _action_handlers;
  }
  const std::vector<std::pair<ActionType, ActionArg>>& flat_action_map() const {
    return _flat_action_map;
  }
  const std::vector<std::string>& flat_action_names() const {
    return _flat_action_names;
  }
  const std::vector<std::vector<int>>& action_arg_to_flat() const {
    return _action_arg_to_flat;
  }
  const std::vector<unsigned char>& max_action_args() const {
    return _max_action_args;
  }
  const std::vector<Agent*>& agents() const {
    return _agents;
  }
  Agent* agent(size_t idx) const {
    return _agents[idx];
  }
  const ObservationEncoder& observation_encoder() const {
    return *_obs_encoder;
  }
  ObservationCoord map_width() const;
  ObservationCoord map_height() const;

  unsigned int current_step = 0;
  unsigned int max_steps = 0;
  bool episode_truncates = false;
  uint64_t initial_grid_hash = 0;

  std::vector<std::string> resource_names;
  std::vector<std::string> object_type_names;

private:
  void init_action_handlers(const GameConfig& game_config);
  void add_agent(Agent* agent);
  void compute_observation(GridCoord observer_row,
                           GridCoord observer_col,
                           ObservationCoord observable_width,
                           ObservationCoord observable_height,
                           size_t agent_idx,
                           ActionType action,
                           ActionArg action_arg);
  void compute_observations(ActionMatrixView actions);
  void run_step(ActionMatrixView actions);
  void handle_invalid_action(size_t agent_idx, const std::string& stat, ActionType type, ActionArg arg);
  void build_flat_action_catalog();

  ObservationCoord obs_width = 0;
  ObservationCoord obs_height = 0;
  GlobalObsConfig _global_obs_config;

  std::unique_ptr<Grid> _grid;
  std::unique_ptr<EventManager> _event_manager;
  std::unique_ptr<ObservationEncoder> _obs_encoder;
  std::unique_ptr<StatsTracker> _stats;

  std::vector<std::unique_ptr<ActionHandler>> _action_handlers;
  std::vector<Agent*> _agents;
  size_t _num_observation_tokens = 0;
  size_t _num_action_handlers = 0;

  std::vector<std::pair<ActionType, ActionArg>> _flat_action_map;
  std::vector<std::string> _flat_action_names;
  std::vector<std::vector<int>> _action_arg_to_flat;
  std::vector<unsigned char> _max_action_args;
  unsigned char _max_action_arg = 0;
  unsigned char _max_action_priority = 0;

  std::unordered_map<unsigned int, float> _group_reward_pct;
  std::unordered_map<unsigned int, unsigned int> _group_sizes;
  std::vector<RewardType> _group_rewards;

  std::vector<ObservationType> _resource_rewards;
  std::vector<bool> _action_success;

  float _resource_loss_prob = 0.0f;
  unsigned int _inventory_regen_interval = 0;
  bool _track_movement_metrics = false;

  std::unique_ptr<class Clipper> _clipper;

  BufferSet _buffers;
  std::mt19937 _rng;
  unsigned int _seed = 0;
};

}  // namespace mettagrid::env

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ENV_METTAGRID_ENGINE_HPP_
