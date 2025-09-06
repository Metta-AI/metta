// ResourceModAction: Consumes and modifies resources with floating-point precision, deterministic rounding
#ifndef ACTIONS_RESOURCE_MOD_HPP_
#define ACTIONS_RESOURCE_MOD_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

struct ResourceModActionConfig : public ActionConfig {
  std::map<InventoryItem, float> consumes;  // Resources consumed from actor (floating point)
  std::map<InventoryItem, float> modifies;  // Resources modified on targets (floating point)
  bool scales;                              // Whether modifications scale by number of targets
  unsigned char agent_radius;               // Radius to affect agents
  unsigned char converter_radius;           // Radius to affect converters

  ResourceModActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources,
                          const std::map<InventoryItem, InventoryQuantity>& consumed_resources,
                          const std::map<InventoryItem, float>& consumes,
                          const std::map<InventoryItem, float>& modifies,
                          bool scales,
                          unsigned char agent_radius,
                          unsigned char converter_radius)
      : ActionConfig(required_resources, consumed_resources),
        consumes(consumes),
        modifies(modifies),
        scales(scales),
        agent_radius(agent_radius),
        converter_radius(converter_radius) {}
};

class ResourceMod : public ActionHandler {
public:
  explicit ResourceMod(const ResourceModActionConfig& cfg, const std::string& action_name = "resource_mod")
      : ActionHandler(cfg, action_name),
        _consumes(cfg.consumes),
        _modifies(cfg.modifies),
        _scales(cfg.scales),
        _agent_radius(cfg.agent_radius),
        _converter_radius(cfg.converter_radius) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  std::map<InventoryItem, float> _consumes;
  std::map<InventoryItem, float> _modifies;
  bool _scales;
  unsigned char _agent_radius;
  unsigned char _converter_radius;

  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    // First consume resources from the actor with floating point precision
    for (const auto& [item, amount] : _consumes) {
      // Convert float to integer delta, rounding probabilistically
      InventoryDelta delta = _float_to_inventory_delta(amount);
      if (delta != 0) {
        InventoryDelta actual_delta = actor->update_inventory(item, -delta);
        if (actual_delta != -delta) {
          // Actor doesn't have enough resources
          return false;
        }
      }
    }

    // Find all agents and converters within radius
    std::vector<Agent*> affected_agents;
    std::vector<Converter*> affected_converters;

    // Scan for agents within agent_radius. This includes the actor.
    GridCoord r_min = std::max(0, static_cast<int>(actor->location.r) - _agent_radius);
    GridCoord r_max =
        std::min(static_cast<int>(_grid->height - 1), static_cast<int>(actor->location.r) + _agent_radius);
    GridCoord c_min = std::max(0, static_cast<int>(actor->location.c) - _agent_radius);
    GridCoord c_max =
        std::min(static_cast<int>(_grid->width - 1), static_cast<int>(actor->location.c) + _agent_radius);

    for (GridCoord r = r_min; r <= r_max; ++r) {
      for (GridCoord c = c_min; c <= c_max; ++c) {
        // Check distance is within radius (Manhattan distance)
        int dr = std::abs(static_cast<int>(r) - static_cast<int>(actor->location.r));
        int dc = std::abs(static_cast<int>(c) - static_cast<int>(actor->location.c));
        if (dr + dc <= _agent_radius) {
          GridLocation loc(r, c, GridLayer::AgentLayer);
          Agent* target = static_cast<Agent*>(_grid->object_at(loc));
          if (target) {
            affected_agents.push_back(target);
          }
        }
      }
    }

    // Scan for converters within converter_radius
    if (_converter_radius > 0) {
      GridCoord r_min = std::max(0, static_cast<int>(actor->location.r) - _converter_radius);
      GridCoord r_max =
          std::min(static_cast<int>(_grid->height - 1), static_cast<int>(actor->location.r) + _converter_radius);
      GridCoord c_min = std::max(0, static_cast<int>(actor->location.c) - _converter_radius);
      GridCoord c_max =
          std::min(static_cast<int>(_grid->width - 1), static_cast<int>(actor->location.c) + _converter_radius);

      for (GridCoord r = r_min; r <= r_max; ++r) {
        for (GridCoord c = c_min; c <= c_max; ++c) {
          // Check distance is within radius (Manhattan distance)
          int dr = std::abs(static_cast<int>(r) - static_cast<int>(actor->location.r));
          int dc = std::abs(static_cast<int>(c) - static_cast<int>(actor->location.c));
          if (dr + dc <= _converter_radius) {
            GridLocation loc(r, c, GridLayer::ObjectLayer);
            GridObject* obj = _grid->object_at(loc);
            if (obj) {
              Converter* converter = dynamic_cast<Converter*>(obj);
              if (converter) {
                affected_converters.push_back(converter);
              }
            }
          }
        }
      }
    }

    // Calculate total affected entities for scaling
    size_t total_affected = affected_agents.size() + affected_converters.size();
    if (total_affected == 0) {
      return true;  // No targets but action succeeds
    }

    // Apply modifications
    for (const auto& [item, base_amount] : _modifies) {
      float amount = base_amount;
      if (_scales && total_affected > 0) {
        // Scale the amount by the number of affected entities
        amount = base_amount / total_affected;
      }

      // Apply to agents
      for (Agent* target : affected_agents) {
        InventoryDelta delta = _float_to_inventory_delta(amount);
        if (delta != 0) {
          target->update_inventory(item, delta);
        }
      }

      // Apply to converters
      for (Converter* target : affected_converters) {
        InventoryDelta delta = _float_to_inventory_delta(amount);
        if (delta != 0) {
          target->update_inventory(item, delta);
        }
      }
    }

    // Log statistics
    actor->stats.add(_action_name + ".agents_affected", affected_agents.size());
    actor->stats.add(_action_name + ".converters_affected", affected_converters.size());

    return true;
  }

private:
  // Convert floating point resource amount to integer delta with probabilistic rounding
  InventoryDelta _float_to_inventory_delta(float amount) const {
    if (amount == 0.0f) {
      return 0;
    }

    // Use probabilistic rounding to handle fractional parts
    // This matches the pattern used for resource loss in mettagrid_c.cpp
    InventoryDelta base_delta = static_cast<InventoryDelta>(std::floor(std::abs(amount)));
    float fractional_part = std::abs(amount) - base_delta;
    
    // With probability equal to the fractional part, round up
    if (fractional_part > 0 && std::generate_canonical<float, 10>(*_rng) < fractional_part) {
      base_delta += 1;
    }
    
    // Apply sign
    return amount > 0 ? base_delta : -base_delta;
  }
};

namespace py = pybind11;

inline void bind_resource_mod_action_config(py::module& m) {
  py::class_<ResourceModActionConfig, ActionConfig, std::shared_ptr<ResourceModActionConfig>>(m,
                                                                                              "ResourceModActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, float>&,
                    const std::map<InventoryItem, float>&,
                    bool,
                    unsigned char,
                    unsigned char>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumes") = std::map<InventoryItem, float>(),
           py::arg("modifies") = std::map<InventoryItem, float>(),
           py::arg("scales") = false,
           py::arg("agent_radius") = 3,
           py::arg("converter_radius") = 2)
      .def_readwrite("consumes", &ResourceModActionConfig::consumes)
      .def_readwrite("modifies", &ResourceModActionConfig::modifies)
      .def_readwrite("scales", &ResourceModActionConfig::scales)
      .def_readwrite("agent_radius", &ResourceModActionConfig::agent_radius)
      .def_readwrite("converter_radius", &ResourceModActionConfig::converter_radius);
}

#endif  // ACTIONS_RESOURCE_MOD_HPP_
