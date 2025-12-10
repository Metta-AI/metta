#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "systems/stats_tracker.hpp"

namespace py = pybind11;

// Holds the cost and object type for a vibe-triggered build
struct VibeBuildEffect {
  // Resource costs to pay (e.g., {energy: 10, carbon: 5})
  std::unordered_map<InventoryItem, InventoryQuantity> cost;
  // Key in objects map to construct
  std::string object_key;

  VibeBuildEffect() = default;
  VibeBuildEffect(const std::unordered_map<InventoryItem, InventoryQuantity>& cost, const std::string& object_key)
      : cost(cost), object_key(object_key) {}
};

struct BuildActionConfig : public ActionConfig {
  // Maps vibe ID to build effects (cost + object to construct)
  std::unordered_map<ObservationType, VibeBuildEffect> vibe_builds;
  bool enabled;
  std::vector<ObservationType> vibes;  // Vibes that trigger this action on move

  BuildActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                    const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources = {},
                    const std::unordered_map<ObservationType, VibeBuildEffect>& vibe_builds = {},
                    bool enabled = true,
                    const std::vector<ObservationType>& vibes = {})
      : ActionConfig(required_resources, consumed_resources),
        vibe_builds(vibe_builds),
        enabled(enabled),
        vibes(vibes) {}
};

// Forward declaration
struct GameConfig;

// Forward declarations
class ObservationEncoder;

class Build : public ActionHandler {
public:
  explicit Build(const BuildActionConfig& cfg,
                 const GameConfig* game_config,
                 StatsTracker* stats_tracker = nullptr,
                 const std::string& action_name = "build")
      : ActionHandler(cfg, action_name),
        _vibe_builds(cfg.vibe_builds),
        _enabled(cfg.enabled),
        _vibes(cfg.vibes),
        _game_config(game_config),
        _stats_tracker(stats_tracker),
        _current_timestep_ptr(nullptr),
        _obs_encoder(nullptr),
        _num_agents(0) {
    priority = 0;  // Lower priority (same as transfer)
  }

  // Set runtime context needed for assembler initialization
  void set_runtime_context(unsigned int* current_timestep_ptr,
                           ObservationEncoder* obs_encoder,
                           unsigned int num_agents) {
    _current_timestep_ptr = current_timestep_ptr;
    _obs_encoder = obs_encoder;
    _num_agents = num_agents;
  }

  std::vector<Action> create_actions() override {
    // Build doesn't create standalone actions - it's triggered by move
    return {};
  }

  // Get vibes that trigger this action on move
  const std::vector<ObservationType>& get_vibes() const {
    return _vibes;
  }

  // Check if the actor's vibe has a build configured
  bool has_build_for_vibe(ObservationType vibe) const {
    return _vibe_builds.find(vibe) != _vibe_builds.end();
  }

  // Try to build after a successful move. Returns true if build succeeded.
  // previous_location: where the agent was before moving (where we'll place the object)
  bool try_build(Agent& actor, const GridLocation& previous_location) {
    const std::string& actor_group = actor.group_name;

    // Debug: track that build was attempted
    actor.stats.incr(_action_prefix(actor_group) + "attempted");
    actor.stats.incr(_action_prefix(actor_group) + "vibe." + std::to_string(actor.vibe));

    auto vibe_it = _vibe_builds.find(actor.vibe);
    if (vibe_it == _vibe_builds.end()) {
      actor.stats.incr(_action_prefix(actor_group) + "failed.no_build_for_vibe");
      return false;  // No build configured for this vibe
    }

    const VibeBuildEffect& effect = vibe_it->second;

    // 1. Check if the previous location is empty (it should be since we just moved)
    if (!_grid->is_empty(previous_location.r, previous_location.c)) {
      return false;  // Location is not empty (shouldn't happen normally)
    }

    // 2. Check if actor has resources to pay the cost
    for (const auto& [resource, amount] : effect.cost) {
      if (actor.inventory.amount(resource) < amount) {
        actor.stats.incr(_action_prefix(actor_group) + "failed.insufficient_resources");
        return false;  // Actor doesn't have enough resources
      }
    }

    // 3. Try to create the object at the previous location
    GridObject* new_object = _create_object(effect.object_key, previous_location);
    if (!new_object) {
      actor.stats.incr(_action_prefix(actor_group) + "failed.object_creation");
      return false;  // Failed to create object
    }

    // 4. Deduct resources from actor
    for (const auto& [resource, amount] : effect.cost) {
      if (amount > 0) {
        InventoryDelta delta = static_cast<InventoryDelta>(-static_cast<int>(amount));
        InventoryDelta actual = actor.inventory.update(resource, delta);
        if (actual != 0) {
          _log_build_cost(actor, resource, -actual);
        }
      }
    }

    // 5. Log success
    actor.stats.incr(_action_prefix(actor_group) + "count");
    actor.stats.incr(_action_prefix(actor_group) + "built." + effect.object_key);

    // Track build stats by object name (agent and game level)
    actor.stats.incr("build." + effect.object_key);
    if (_stats_tracker) {
      _stats_tracker->incr("build." + effect.object_key);
    }

    return true;
  }

protected:
  std::unordered_map<ObservationType, VibeBuildEffect> _vibe_builds;
  bool _enabled;
  std::vector<ObservationType> _vibes;
  const GameConfig* _game_config;
  StatsTracker* _stats_tracker;
  // Runtime context for assembler initialization
  unsigned int* _current_timestep_ptr;
  ObservationEncoder* _obs_encoder;
  unsigned int _num_agents;

  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Build is not called directly as an action
    (void)actor;
    (void)arg;
    return false;
  }

private:
  std::string _action_prefix(const std::string& group) const {
    return "action." + _action_name + "." + group + ".";
  }

  void _log_build_cost(Agent& actor, InventoryItem item, InventoryDelta amount) const {
    const std::string& actor_group = actor.group_name;
    const std::string item_name = actor.stats.resource_name(item);

    actor.stats.add(_action_prefix(actor_group) + "cost." + item_name, amount);
  }

  // Create the object at the given location
  // Returns the created object or nullptr on failure
  GridObject* _create_object(const std::string& object_key, const GridLocation& location);
};

inline void bind_vibe_build_effect(py::module& m) {
  py::class_<VibeBuildEffect>(m, "VibeBuildEffect")
      .def(py::init<>())
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&, const std::string&>(),
           py::arg("cost") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("object_key") = std::string())
      .def_readwrite("cost", &VibeBuildEffect::cost)
      .def_readwrite("object_key", &VibeBuildEffect::object_key);
}

inline void bind_build_action_config(py::module& m) {
  py::class_<BuildActionConfig, ActionConfig, std::shared_ptr<BuildActionConfig>>(m, "BuildActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<ObservationType, VibeBuildEffect>&,
                    bool,
                    const std::vector<ObservationType>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("vibe_builds") = std::unordered_map<ObservationType, VibeBuildEffect>(),
           py::arg("enabled") = true,
           py::arg("vibes") = std::vector<ObservationType>())
      .def_readwrite("vibe_builds", &BuildActionConfig::vibe_builds)
      .def_readwrite("enabled", &BuildActionConfig::enabled)
      .def_readwrite("vibes", &BuildActionConfig::vibes);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_BUILD_HPP_
