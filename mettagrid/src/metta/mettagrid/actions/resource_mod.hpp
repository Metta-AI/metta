// ResourceModAction: Probabilistically consumes and modifies resources based on probability values
#ifndef ACTIONS_RESOURCE_MOD_HPP_
#define ACTIONS_RESOURCE_MOD_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

struct ResourceModActionConfig : public ActionConfig {
  std::map<InventoryItem, float> consumes;  // Probability of consuming resources from actor
  std::map<InventoryItem, float> modifies;  // Probability of modifying resources on targets
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
        _converter_radius(cfg.converter_radius),
        _original_consumed_resources(cfg.consumed_resources) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  std::map<InventoryItem, float> _consumes;
  std::map<InventoryItem, float> _modifies;
  bool _scales;
  unsigned char _agent_radius;
  unsigned char _converter_radius;

  // Store original consumed_resources to restore on failure
  std::map<InventoryItem, InventoryQuantity> _original_consumed_resources;

  // Helper to compute probabilistic delta from float amount using integer+fractional semantics
  InventoryDelta compute_probabilistic_delta(float amount, std::mt19937* rng) const {
    float abs_amount = std::abs(amount);
    if (abs_amount == 0.0f) return 0;

    // Integer part applied deterministically
    int integer_part = static_cast<int>(std::floor(abs_amount));
    float fractional_part = abs_amount - integer_part;

    InventoryDelta delta = integer_part;

    // Fractional part applied probabilistically
    if (fractional_part > 0.0f && std::generate_canonical<float, 10>(*rng) < fractional_part) {
      delta += 1;
    }

    // Apply sign
    if (amount < 0) {
      delta = -delta;
    }

    return delta;
  }

  // Template helper to find objects within radius using Manhattan distance
  template <typename T>
  std::vector<T*> find_objects_in_radius(Agent* actor, unsigned char radius, ObservationType layer) const {
    std::vector<T*> objects;

    GridCoord r_min = std::max(0, static_cast<int>(actor->location.r) - radius);
    GridCoord r_max = std::min(static_cast<int>(_grid->height - 1), static_cast<int>(actor->location.r) + radius);
    GridCoord c_min = std::max(0, static_cast<int>(actor->location.c) - radius);
    GridCoord c_max = std::min(static_cast<int>(_grid->width - 1), static_cast<int>(actor->location.c) + radius);

    for (GridCoord r = r_min; r <= r_max; ++r) {
      for (GridCoord c = c_min; c <= c_max; ++c) {
        // Check distance is within radius (Manhattan distance)
        int dr = std::abs(static_cast<int>(r) - static_cast<int>(actor->location.r));
        int dc = std::abs(static_cast<int>(c) - static_cast<int>(actor->location.c));
        if (dr + dc <= radius) {
          GridLocation loc(r, c, layer);
          GridObject* obj = _grid->object_at(loc);
          if (obj) {
            // For Agents we can cast directly, for Converters we need dynamic_cast
            T* target = nullptr;
            if (layer == GridLayer::AgentLayer) {
              target = static_cast<T*>(obj);
            } else {
              target = dynamic_cast<T*>(obj);
            }
            if (target) {
              objects.push_back(target);
            }
          }
        }
      }
    }
    return objects;
  }

  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    // Use local variable for thread safety - don't modify member during execution
    std::map<InventoryItem, InventoryQuantity> consumed_resources = _original_consumed_resources;

    // Find all agents and converters within radius
    std::vector<Agent*> affected_agents = find_objects_in_radius<Agent>(actor, _agent_radius, ::GridLayer::AgentLayer);
    std::vector<Converter*> affected_converters =
        find_objects_in_radius<Converter>(actor, _converter_radius, ::GridLayer::ObjectLayer);

    // Calculate total affected entities for scaling
    size_t total_affected = affected_agents.size() + affected_converters.size();

    // Gate consumption on having targets (if modifies is not empty)
    if (!_modifies.empty() && total_affected == 0) {
      // No targets to modify - fail without consuming resources
      return false;
    }

    // Pre-compute all consumption deltas based on probabilities
    std::map<InventoryItem, InventoryDelta> consume_deltas;
    for (const auto& [item, amount] : _consumes) {
      InventoryDelta delta = compute_probabilistic_delta(amount, _rng);
      if (delta != 0) {
        consume_deltas[item] = delta;
      }
    }

    // Add probabilistic consumes to local consumed_resources for base class to handle
    // This ensures all consumption happens at the same time (after success)
    for (const auto& [item, delta] : consume_deltas) {
      if (delta > 0) {
        consumed_resources[item] += static_cast<InventoryQuantity>(delta);
      }
    }

    // Pre-compute all modification deltas for each target
    struct ModificationDelta {
      GridObject* target;
      InventoryItem item;
      InventoryDelta delta;
    };
    std::vector<ModificationDelta> modification_deltas;

    for (const auto& [item, base_amount] : _modifies) {
      float amount = base_amount;
      if (_scales && total_affected > 0) {
        // Scale the amount by the number of affected entities
        amount = base_amount / total_affected;
      }

      // Compute deltas for agents
      for (Agent* target : affected_agents) {
        InventoryDelta delta = compute_probabilistic_delta(amount, _rng);
        if (delta != 0) {
          modification_deltas.push_back({target, item, delta});
        }
      }

      // Compute deltas for converters
      for (Converter* target : affected_converters) {
        InventoryDelta delta = compute_probabilistic_delta(amount, _rng);
        if (delta != 0) {
          modification_deltas.push_back({target, item, delta});
        }
      }
    }

    // Validate consumed_resources (which now includes probabilistic consumes)
    for (const auto& [item, required_amount] : consumed_resources) {
      if (required_amount > 0) {
        auto inv_it = actor->inventory.find(item);
        InventoryQuantity current_amount = (inv_it != actor->inventory.end()) ? inv_it->second : 0;

        if (current_amount < required_amount) {
          // Actor doesn't have enough resources - fail without any state changes
          return false;
        }
      }
    }

    // Apply modifications to targets
    for (const auto& mod : modification_deltas) {
      if (Agent* agent = dynamic_cast<Agent*>(mod.target)) {
        agent->update_inventory(mod.item, mod.delta);
      } else if (Converter* converter = dynamic_cast<Converter*>(mod.target)) {
        converter->update_inventory(mod.item, mod.delta);
      }
    }

    // Log statistics
    actor->stats.add(_action_name + ".agents_affected", affected_agents.size());
    actor->stats.add(_action_name + ".converters_affected", affected_converters.size());

    // Update member variable only at the end for base class to use
    _consumed_resources = consumed_resources;
    return true;
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
