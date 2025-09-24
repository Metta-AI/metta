// modify_target.hpp - Simple action to modify resources of a target at a position
#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <map>
#include <string>
#include <stdexcept>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "core/types.hpp"

struct ModifyTargetConfig : public ActionConfig {
  std::map<InventoryItem, InventoryProbability> modifies;
  uint8_t agent_radius = 3;
  uint8_t converter_radius = 2;
  bool scales = false;

  ModifyTargetConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources = {},
                     const std::map<InventoryItem, InventoryProbability>& consumed_resources = {},
                     const std::map<InventoryItem, InventoryProbability>& modifies = {},
                     uint8_t agent_radius = 3,
                     uint8_t converter_radius = 2,
                     bool scales = false)
      : ActionConfig(required_resources, consumed_resources),
        modifies(modifies),
        agent_radius(agent_radius),
        converter_radius(converter_radius),
        scales(scales) {
    // Validate modifies values for finiteness and magnitude
    for (const auto& [item, value] : modifies) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("ModifyTargetConfig: modifies values must be finite");
      }
      // Check that absolute value when rounded up doesn't exceed 255
      // (since inventories are uint8_t and deltas are int16_t)
      if (std::ceil(std::abs(value)) > 255) {
        throw std::invalid_argument("ModifyTargetConfig: modifies values must have |ceil(value)| <= 255");
      }
    }
  }
};

class ModifyTarget : public ActionHandler {
public:
  explicit ModifyTarget(const ModifyTargetConfig& cfg, const std::string& name = "modify_target")
      : ActionHandler(cfg, name),
        _modifies(cfg.modifies),
        _agent_radius(cfg.agent_radius),
        _converter_radius(cfg.converter_radius),
        _scales(cfg.scales) {}

  unsigned char max_arg() const override {
    // No arg needed for radius-based area-of-effect action
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Collect all targets within configured radii
    std::vector<Agent*> agent_targets;
    std::vector<Converter*> converter_targets;

    // Find all agents within agent_radius
    if (_agent_radius > 0) {
      for (int dr = -static_cast<int>(_agent_radius); dr <= static_cast<int>(_agent_radius); ++dr) {
        for (int dc = -static_cast<int>(_agent_radius); dc <= static_cast<int>(_agent_radius); ++dc) {
          // Calculate Manhattan distance
          if (std::abs(dr) + std::abs(dc) > static_cast<int>(_agent_radius)) {
            continue;
          }

          int target_row = static_cast<int>(actor->location.r) + dr;
          int target_col = static_cast<int>(actor->location.c) + dc;

          // Check bounds
          if (target_row < 0 || target_row >= static_cast<int>(_grid->height) ||
              target_col < 0 || target_col >= static_cast<int>(_grid->width)) {
            continue;
          }

          GridLocation agent_loc(target_row, target_col, GridLayer::AgentLayer);
          GridObject* obj = _grid->object_at(agent_loc);
          if (obj != nullptr) {
            Agent* agent = static_cast<Agent*>(obj);
            // Include all agents within radius, including self
            agent_targets.push_back(agent);
          }
        }
      }
    }

    // Find all converters within converter_radius
    if (_converter_radius > 0) {
      for (int dr = -static_cast<int>(_converter_radius); dr <= static_cast<int>(_converter_radius); ++dr) {
        for (int dc = -static_cast<int>(_converter_radius); dc <= static_cast<int>(_converter_radius); ++dc) {
          // Calculate Manhattan distance
          if (std::abs(dr) + std::abs(dc) > static_cast<int>(_converter_radius)) {
            continue;
          }

          int target_row = static_cast<int>(actor->location.r) + dr;
          int target_col = static_cast<int>(actor->location.c) + dc;

          // Check bounds
          if (target_row < 0 || target_row >= static_cast<int>(_grid->height) ||
              target_col < 0 || target_col >= static_cast<int>(_grid->width)) {
            continue;
          }

          GridLocation obj_loc(target_row, target_col, GridLayer::ObjectLayer);
          GridObject* obj = _grid->object_at(obj_loc);
          if (obj != nullptr) {
            Converter* converter = dynamic_cast<Converter*>(obj);
            if (converter != nullptr) {
              converter_targets.push_back(converter);
            }
          }
        }
      }
    }

    // If no targets found, action fails
    size_t total_targets = agent_targets.size() + converter_targets.size();
    if (total_targets == 0) {
      return false;
    }

    // Track whether any changes were actually applied
    bool applied_any_change = false;

    // Apply modifications to all targets
    for (const auto& [item, amount] : _modifies) {
      InventoryProbability effective_amount = amount;

      // Scale the effect if scales is enabled
      if (_scales && total_targets > 0) {
        effective_amount = amount / static_cast<float>(total_targets);
      }

      // Apply to all agent targets with per-target randomness
      for (Agent* agent : agent_targets) {
        // Sample independently for each target for variance independence
        InventoryDelta delta = compute_probabilistic_delta(effective_amount);
        if (delta != 0) {
          agent->update_inventory(item, delta);
          applied_any_change = true;
        }
      }

      // Apply to all converter targets with per-target randomness
      for (Converter* converter : converter_targets) {
        // Sample independently for each target for variance independence
        InventoryDelta delta = compute_probabilistic_delta(effective_amount);
        if (delta != 0) {
          converter->update_inventory(item, delta);
          applied_any_change = true;
        }
      }
    }

    // Return true only if changes were actually applied
    return applied_any_change;
  }

private:
  std::map<InventoryItem, InventoryProbability> _modifies;
  uint8_t _agent_radius;
  uint8_t _converter_radius;
  bool _scales;
};

namespace py = pybind11;

inline void bind_modify_target_config(py::module& m) {
  py::class_<ModifyTargetConfig, ActionConfig, std::shared_ptr<ModifyTargetConfig>>(m, "ModifyTargetConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryProbability>&,
                    const std::map<InventoryItem, InventoryProbability>&,
                    uint8_t,
                    uint8_t,
                    bool>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryProbability>(),
           py::arg("modifies") = std::map<InventoryItem, InventoryProbability>(),
           py::arg("agent_radius") = 3,
           py::arg("converter_radius") = 2,
           py::arg("scales") = false)
      .def_readwrite("modifies", &ModifyTargetConfig::modifies)
      .def_readwrite("agent_radius", &ModifyTargetConfig::agent_radius)
      .def_readwrite("converter_radius", &ModifyTargetConfig::converter_radius)
      .def_readwrite("scales", &ModifyTargetConfig::scales);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MODIFY_TARGET_HPP_
