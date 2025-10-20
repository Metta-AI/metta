#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_RESOURCE_MOD_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_RESOURCE_MOD_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "core/grid.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"

struct ResourceModConfig : public ActionConfig {
  std::unordered_map<InventoryItem, InventoryProbability> modifies;
  GridCoord agent_radius;
  GridCoord converter_radius;
  bool scales;

  ResourceModConfig(const std::unordered_map<InventoryItem, InventoryQuantity>& required_resources = {},
                    const std::unordered_map<InventoryItem, InventoryProbability>& consumed_resources = {},
                    const std::unordered_map<InventoryItem, InventoryProbability>& modifies = {},
                    GridCoord agent_radius = 0,
                    GridCoord converter_radius = 0,
                    bool scales = false)
      : ActionConfig(required_resources, consumed_resources),
        modifies(modifies),
        agent_radius(agent_radius),
        converter_radius(converter_radius),
        scales(scales) {
    // Validate modifies values for finiteness and magnitude
    for (const auto& [item, value] : modifies) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("ResourceModConfig: modifies values must be finite");
      }
      // Check that absolute value when rounded up doesn't exceed 255
      // (since inventories are uint8_t and deltas are int16_t)
      if (std::ceil(std::abs(value)) > 255) {
        throw std::invalid_argument("ResourceModConfig: modifies values must have |ceil(value)| <= 255");
      }
    }
  }
};

class ResourceMod : public ActionHandler {
public:
  explicit ResourceMod(const ResourceModConfig& cfg, const std::string& name = "resource_mod")
      : ActionHandler(cfg, name),
        _modifies(cfg.modifies),
        _agent_radius(cfg.agent_radius),
        _converter_radius(cfg.converter_radius),
        _scales(cfg.scales) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  std::string variant_name(ActionArg arg) const override {
    if (arg == 0) {
      return action_name();
    }
    return ActionHandler::variant_name(arg);
  }

  bool _handle_action(Agent& actor, ActionArg /* arg */) override {
    // Center AoE on actor's position
    int center_row = static_cast<int>(actor.location.r);
    int center_col = static_cast<int>(actor.location.c);

    std::vector<Agent*> affected_agents;
    std::vector<Converter*> affected_converters;

    // Find all agents within agent_radius of the center
    // radius=0 means skip agents entirely
    if (_agent_radius > 0) {
      for (int dr = -static_cast<int>(_agent_radius); dr <= static_cast<int>(_agent_radius); ++dr) {
        for (int dc = -static_cast<int>(_agent_radius); dc <= static_cast<int>(_agent_radius); ++dc) {
          // Manhattan distance
          if (std::abs(dr) + std::abs(dc) > static_cast<int>(_agent_radius)) {
            continue;
          }

          int target_row = center_row + dr;
          int target_col = center_col + dc;

          if (target_row >= 0 && target_row < static_cast<int>(_grid->height) && target_col >= 0 &&
              target_col < static_cast<int>(_grid->width)) {
            GridLocation loc(target_row, target_col, GridLayer::AgentLayer);
            GridObject* obj = _grid->object_at(loc);
            if (obj != nullptr) {
              Agent* agent = static_cast<Agent*>(obj);
              affected_agents.push_back(agent);
            }
          }
        }
      }
    }

    // Find all converters within converter_radius of the center
    // radius=0 means skip converters entirely
    if (_converter_radius > 0) {
      for (int dr = -static_cast<int>(_converter_radius); dr <= static_cast<int>(_converter_radius); ++dr) {
        for (int dc = -static_cast<int>(_converter_radius); dc <= static_cast<int>(_converter_radius); ++dc) {
          // Manhattan distance
          if (std::abs(dr) + std::abs(dc) > static_cast<int>(_converter_radius)) {
            continue;
          }

          int target_row = center_row + dr;
          int target_col = center_col + dc;

          if (target_row >= 0 && target_row < static_cast<int>(_grid->height) && target_col >= 0 &&
              target_col < static_cast<int>(_grid->width)) {
            GridLocation loc(target_row, target_col, GridLayer::ObjectLayer);
            GridObject* obj = _grid->object_at(loc);
            if (obj != nullptr) {
              Converter* converter = dynamic_cast<Converter*>(obj);
              if (converter != nullptr) {
                affected_converters.push_back(converter);
              }
            }
          }
        }
      }
    }

    // No targets found - still consume resources and succeed
    if (affected_agents.empty() && affected_converters.empty()) {
      return true;
    }

    // Calculate total number of affected targets for scaling
    size_t total_affected = affected_agents.size() + affected_converters.size();

    // Apply modifications to all targets
    for (const auto& [item, amount] : _modifies) {
      InventoryProbability effective_amount = amount;

      // Scale by number of targets if scales flag is true
      if (_scales && total_affected > 1) {
        effective_amount = amount / static_cast<float>(total_affected);
      }

      // Apply to agents - compute delta independently for each target
      for (Agent* agent : affected_agents) {
        InventoryDelta delta = compute_probabilistic_delta(effective_amount);
        if (delta != 0) {
          agent->update_inventory(item, delta);
        }
      }

      // Apply to converters - compute delta independently for each target
      for (Converter* converter : affected_converters) {
        InventoryDelta delta = compute_probabilistic_delta(effective_amount);
        if (delta != 0) {
          converter->update_inventory(item, delta);
        }
      }
    }

    return true;
  }

private:
  std::unordered_map<InventoryItem, InventoryProbability> _modifies;
  GridCoord _agent_radius;
  GridCoord _converter_radius;
  bool _scales;
};

namespace py = pybind11;

inline void bind_resource_mod_config(py::module& m) {
  py::class_<ResourceModConfig, ActionConfig, std::shared_ptr<ResourceModConfig>>(m, "ResourceModConfig")
      .def(py::init<const std::unordered_map<InventoryItem, InventoryQuantity>&,
                    const std::unordered_map<InventoryItem, InventoryProbability>&,
                    const std::unordered_map<InventoryItem, InventoryProbability>&,
                    GridCoord,
                    GridCoord,
                    bool>(),
           py::arg("required_resources") = std::unordered_map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::unordered_map<InventoryItem, InventoryProbability>(),
           py::arg("modifies") = std::unordered_map<InventoryItem, InventoryProbability>(),
           py::arg("agent_radius") = 0,
           py::arg("converter_radius") = 0,
           py::arg("scales") = false)
      .def_readwrite("modifies", &ResourceModConfig::modifies)
      .def_readwrite("agent_radius", &ResourceModConfig::agent_radius)
      .def_readwrite("converter_radius", &ResourceModConfig::converter_radius)
      .def_readwrite("scales", &ResourceModConfig::scales);
}

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_RESOURCE_MOD_HPP_
