// action_handler.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cmath>
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
  std::unordered_map<InventoryItem, InventoryProbability> consumed_resources;

  ActionConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
               const std::unordered_map<InventoryItem, InventoryProbability>& consumed_resources = {})
      : required_resources(required_resources), consumed_resources(consumed_resources) {}

  virtual ~ActionConfig() {}
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
    // Validate consumed_resources values are non-negative and finite
    for (const auto& [item, probability] : _consumed_resources) {
      if (!std::isfinite(probability) || probability < 0.0f) {
        throw std::runtime_error("Consumed resources must be non-negative and finite. Item: " + std::to_string(item) +
                                 " has invalid value: " + std::to_string(probability));
      }

      // Guard against overflow when casting to uint8_t
      float ceiled = std::ceil(probability);
      if (ceiled > 255.0f) {
        throw std::runtime_error("Consumed resources ceiling exceeds uint8_t max (255). Item: " + std::to_string(item) +
                                 " has ceiling: " + std::to_string(ceiled));
      }
    }

    // Check that required_resources has all items from consumed_resources
    for (const auto& [item, probability] : _consumed_resources) {
      auto required_it = _required_resources.find(item);
      if (required_it == _required_resources.end()) {
        throw std::runtime_error("Consumed resource item " + std::to_string(item) + " not found in required resources");
      }

      // Validate required >= ceil(consumed)
      InventoryQuantity max_consumption = static_cast<InventoryQuantity>(std::ceil(probability));
      if (required_it->second < max_consumption) {
        throw std::runtime_error("Required resources must be >= ceil(consumed resources). Item: " +
                                 std::to_string(item) + " required: " + std::to_string(required_it->second) +
                                 " < ceil(consumed): " + std::to_string(max_consumption));
      }
    }
  }

  virtual ~ActionHandler() {}

  void init(Grid* grid, std::mt19937* rng) {
    this->_grid = grid;
    _rng = rng;
  }

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
    actor.prev_action_name = _action_name;
    actor.prev_location = actor.location;

    // Track success/failure
    if (success) {
      actor.stats.incr("action." + _action_name + ".success");
      for (const auto& [item, amount] : _consumed_resources) {
        InventoryDelta delta = compute_probabilistic_delta(-amount);
        if (delta != 0) {
          [[maybe_unused]] InventoryDelta actual_delta = actor.update_inventory(item, delta);
          // We consume resources after the action succeeds, but in the future we might have an action that uses the
          // resource. This check will catch that.
          assert(actual_delta == delta);
        }
      }
    } else {
      actor.stats.incr("action." + _action_name + ".failed");
      actor.stats.incr("action.failure_penalty");
      *actor.reward -= actor.action_failure_penalty;
    }

    return success;
  }

  virtual unsigned char max_arg() const {
    return 0;
  }

  std::string action_name() const {
    return _action_name;
  }

  virtual std::string variant_name(ActionArg arg) const {
    if (max_arg() == 0) {
      return _action_name;
    }
    return _action_name + "_" + std::to_string(static_cast<int>(arg));
  }

protected:
  virtual bool _handle_action(Agent& actor, ActionArg arg) = 0;

  InventoryDelta compute_probabilistic_delta(InventoryProbability amount) const {
    if (_rng == nullptr) {
      throw std::runtime_error("RNG not initialized. Call init() before using compute_probabilistic_delta");
    }
    InventoryProbability magnitude = std::fabs(amount);
    InventoryQuantity integer_part = static_cast<InventoryQuantity>(std::floor(magnitude));
    InventoryProbability fractional_part = magnitude - static_cast<InventoryProbability>(integer_part);
    InventoryDelta delta = static_cast<InventoryDelta>(integer_part);
    if (fractional_part > 0.0f) {
      // use 10 bits of randomness for better performance
      float sample = std::generate_canonical<float, 10>(*_rng);
      if (sample < fractional_part) {
        // a non-zero fractional component means there is a chance to increase the delta by 1. for example an
        // amount of 4.1 means that the delta will be 4 90% of the time and 5 10% of the time.
        delta = static_cast<InventoryDelta>(delta + 1);
      }
    }
    if (amount < 0.0f) {
      // restore the original sign if needed
      delta = static_cast<InventoryDelta>(-delta);
    }
    return delta;
  }

  std::string _action_name;
  std::unordered_map<InventoryItem, InventoryQuantity> _required_resources;
  std::unordered_map<InventoryItem, InventoryProbability> _consumed_resources;
  std::mt19937* _rng{};
};

namespace py = pybind11;

inline void bind_action_config(py::module& m) {
  py::class_<ActionConfig, std::shared_ptr<ActionConfig>>(m, "ActionConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryProbability>&>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryProbability>())
      .def_readwrite("required_resources", &ActionConfig::required_resources)
      .def_readwrite("consumed_resources", &ActionConfig::consumed_resources);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
