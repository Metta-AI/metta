// action_handler.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

struct ActionConfig {
  std::unordered_map<InventoryItem, InventoryQuantity> required_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> consumed_resources;

  ActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
               const std::unordered_map<InventoryItem, InventoryQuantity>& consumed_resources = {})
      : required_resources(required_resources), consumed_resources(consumed_resources) {}

  virtual ~ActionConfig() {}
};

// Forward declaration
class ActionHandler;

// Action represents a specific action variant (e.g., move_north, attack_0, etc.)
class Action {
public:
  Action(ActionHandler* handler, const std::string& name, ActionArg arg) : _handler(handler), _name(name), _arg(arg) {}

  bool handle(Agent& actor);

  std::string name() const {
    return _name;
  }

  ActionArg arg() const {
    return _arg;
  }

  ActionHandler* handler() const {
    return _handler;
  }

private:
  ActionHandler* _handler;
  std::string _name;
  ActionArg _arg;
};

class ActionHandler {
public:
  unsigned char priority;
  Grid* _grid{};

  ActionHandler(const ActionConfig& cfg, const std::string& action_name)
      : priority(0),
        _action_name(action_name),
        _required_resources(cfg.required_resources),
        _consumed_resources(cfg.consumed_resources) {
    // Check that required_resources has all items from consumed_resources
    for (const auto& [item, amount] : _consumed_resources) {
      auto required_it = _required_resources.find(item);
      if (required_it == _required_resources.end()) {
        throw std::runtime_error("Consumed resource item " + std::to_string(item) + " not found in required resources");
      }

      // Validate required >= consumed
      if (required_it->second < amount) {
        throw std::runtime_error("Required resources must be >= consumed resources. Item: " + std::to_string(item) +
                                 " required: " + std::to_string(required_it->second) +
                                 " < consumed: " + std::to_string(amount));
      }
    }
  }

  virtual ~ActionHandler() {}

  void init(Grid* grid, std::mt19937* rng) {
    this->_grid = grid;
    _rng = rng;

    // Create actions after construction, when the derived class vtable is set up
    if (_actions.empty()) {
      _actions = create_actions();
    }
  }

  // Returns true if the action was executed, false otherwise. In particular, a result of false should have no impact
  // on the environment, and should imply that the agent effectively took a noop action.
  bool handle_action(Agent& actor, ActionArg arg) {
    // Handle frozen status
    if (actor.frozen != 0) {
      actor.stats.incr("status.frozen.ticks");
      actor.stats.incr("status.frozen.ticks." + actor.group_name);
      if (actor.frozen > 0) {
        actor.frozen -= 1;
      }
      return false;
    }

    bool has_needed_resources = true;
    for (const auto& [item, amount] : _required_resources) {
      if (actor.inventory.amount(item) < amount) {
        has_needed_resources = false;
        break;
      }
    }

    // Execute the action
    bool success = has_needed_resources && _handle_action(actor, arg);

    // The intention here is to provide a metric that reports when an agent has stayed in one location for a long
    // period, perhaps spinning in circles. We think this could be a good indicator that a policy has collapsed.
    if (actor.location == actor.prev_location) {
      actor.steps_without_motion += 1;
      if (actor.steps_without_motion > actor.stats.get("status.max_steps_without_motion")) {
        actor.stats.set("status.max_steps_without_motion", actor.steps_without_motion);
      }
    } else {
      actor.steps_without_motion = 0;
    }

    // Update tracking for this agent
    actor.prev_location = actor.location;

    // Track success/failure
    if (success) {
      actor.stats.incr("action." + _action_name + ".success");
      for (const auto& [item, amount] : _consumed_resources) {
        if (amount > 0) {
          InventoryDelta delta = static_cast<InventoryDelta>(-static_cast<int>(amount));
          [[maybe_unused]] InventoryDelta actual_delta = actor.inventory.update(item, delta);
          // We consume resources after the action succeeds, but in the future we might have an action that uses the
          // resource. This check will catch that.
          assert(actual_delta == delta);
        }
      }
    } else {
      actor.stats.incr("action." + _action_name + ".failed");
      actor.stats.incr("action.failed");
    }

    return success;
  }

  std::string action_name() const {
    return _action_name;
  }

  virtual std::string variant_name(ActionArg arg) const {
    return _action_name + "_" + std::to_string(static_cast<int>(arg));
  }

  // Get the actions for this handler
  const std::vector<Action>& actions() const {
    return _actions;
  }

protected:
  // Subclasses override this to create their specific action instances
  virtual std::vector<Action> create_actions() = 0;

  virtual bool _handle_action(Agent& actor, ActionArg arg) = 0;

  std::string _action_name;
  std::unordered_map<InventoryItem, InventoryQuantity> _required_resources;
  std::unordered_map<InventoryItem, InventoryQuantity> _consumed_resources;
  std::mt19937* _rng{};
  std::vector<Action> _actions;
};

// Implement Action::handle() inline after ActionHandler is fully defined
inline bool Action::handle(Agent& actor) {
  return _handler->handle_action(actor, _arg);
}

namespace py = pybind11;

inline void bind_action_config(py::module& m) {
  py::class_<ActionConfig, std::shared_ptr<ActionConfig>>(m, "ActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryQuantity>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryQuantity>())
      .def_readwrite("required_resources", &ActionConfig::required_resources)
      .def_readwrite("consumed_resources", &ActionConfig::consumed_resources);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
