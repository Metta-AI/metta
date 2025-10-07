// action_handler.hpp
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cmath>
#include <map>
#include <random>
#include <string>

#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"

struct ActionConfig {
  std::map<InventoryItem, InventoryQuantity> required_resources;
  std::map<InventoryItem, InventoryProbability> consumed_resources;

  ActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources = {},
               const std::map<InventoryItem, InventoryProbability>& consumed_resources = {})
      : required_resources(required_resources), consumed_resources(consumed_resources) {}

  virtual ~ActionConfig() = default;
};

class ActionHandler {
public:
  unsigned char priority;

  ActionHandler(const ActionConfig& cfg, const std::string& action_name)
      : priority(0),
        _action_name(action_name),
        _required_resources(cfg.required_resources),
        _consumed_resources(cfg.consumed_resources) {
    for (const auto& [item, probability] : _consumed_resources) {
      if (!std::isfinite(probability) || probability < 0.0f) {
        throw std::runtime_error("Consumed resources must be non-negative and finite. Item: " + std::to_string(item) +
                                 " has invalid value: " + std::to_string(probability));
      }

      float ceiled = std::ceil(probability);
      if (ceiled > 255.0f) {
        throw std::runtime_error("Consumed resources ceiling exceeds uint8_t max (255). Item: " + std::to_string(item) +
                                 " has ceiling: " + std::to_string(ceiled));
      }
    }

    for (const auto& [item, probability] : _consumed_resources) {
      auto required_it = _required_resources.find(item);
      if (required_it == _required_resources.end()) {
        throw std::runtime_error("Consumed resource item " + std::to_string(item) + " not found in required resources");
      }

      InventoryQuantity max_consumption = static_cast<InventoryQuantity>(std::ceil(probability));
      if (required_it->second < max_consumption) {
        throw std::runtime_error("Required resources must be >= ceil(consumed resources). Item: " +
                                 std::to_string(item) + " required: " + std::to_string(required_it->second) +
                                 " < ceil(consumed): " + std::to_string(max_consumption));
      }
    }
  }

  virtual ~ActionHandler() = default;

  void init(Grid* grid, std::mt19937* rng) {
    _grid = grid;
    _rng = rng;
  }

  bool handle_action(Agent& actor) {
    if (actor.frozen != 0) {
      actor.stats.incr("status.frozen.ticks");
      actor.stats.incr("status.frozen.ticks." + actor.group_name);
      if (actor.frozen > 0) {
        actor.frozen -= 1;
      }
      return false;
    }

    for (const auto& [item, amount] : _required_resources) {
      if (actor.inventory.amount(item) < amount) {
        actor.stats.incr("action." + _action_name + ".failed");
        actor.stats.incr("action.failure_penalty");
        *actor.reward -= actor.action_failure_penalty;
        return false;
      }
    }

    bool success = _handle_action(actor);

    if (actor.location == actor.prev_location) {
      actor.steps_without_motion += 1;
      if (actor.steps_without_motion > actor.stats.get("status.max_steps_without_motion")) {
        actor.stats.set("status.max_steps_without_motion", actor.steps_without_motion);
      }
    } else {
      actor.steps_without_motion = 0;
    }

    actor.prev_action_name = _action_name;
    actor.prev_location = actor.location;

    if (success) {
      actor.stats.incr("action." + _action_name + ".success");
      for (const auto& [item, amount] : _consumed_resources) {
        InventoryDelta delta = compute_probabilistic_delta(-amount);
        if (delta != 0) {
          [[maybe_unused]] InventoryDelta actual_delta = actor.update_inventory(item, delta);
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

  const std::string& action_name() const {
    return _action_name;
  }

protected:
  Grid& grid() const {
    return *_grid;
  }

  std::mt19937& rng() const {
    return *_rng;
  }

  virtual bool _handle_action(Agent& actor) = 0;

  InventoryDelta compute_probabilistic_delta(InventoryProbability amount) const {
    if (_rng == nullptr) {
      throw std::runtime_error("RNG not initialized. Call init() before using compute_probabilistic_delta");
    }
    InventoryProbability magnitude = std::fabs(amount);
    InventoryQuantity integer_part = static_cast<InventoryQuantity>(std::floor(magnitude));
    InventoryProbability fractional_part = magnitude - static_cast<InventoryProbability>(integer_part);
    InventoryDelta delta = static_cast<InventoryDelta>(integer_part);
    if (fractional_part > 0.0f) {
      float sample = std::generate_canonical<float, 10>(*_rng);
      if (sample < fractional_part) {
        delta = static_cast<InventoryDelta>(delta + 1);
      }
    }
    if (amount < 0.0f) {
      delta = static_cast<InventoryDelta>(-delta);
    }
    return delta;
  }

protected:
  Grid* _grid{};
  std::mt19937* _rng{};

private:
  std::string _action_name;
  std::map<InventoryItem, InventoryQuantity> _required_resources;
  std::map<InventoryItem, InventoryProbability> _consumed_resources;
};

namespace py = pybind11;

inline void bind_action_config(py::module& m) {
  py::class_<ActionConfig, std::shared_ptr<ActionConfig>>(m, "ActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryProbability>&>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryProbability>())
      .def_readwrite("required_resources", &ActionConfig::required_resources)
      .def_readwrite("consumed_resources", &ActionConfig::consumed_resources);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ACTION_HANDLER_HPP_
