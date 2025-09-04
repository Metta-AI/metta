#ifndef ACTIONS_PUT_RECIPE_ITEMS_HPP_
#define ACTIONS_PUT_RECIPE_ITEMS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"

struct PutItemsActionConfig : public ActionConfig {
  bool facing_required;

  PutItemsActionConfig(const std::map<InventoryItem, InventoryQuantity>& required_resources = {},
                       const std::map<InventoryItem, InventoryQuantity>& consumed_resources = {},
                       unsigned char priority = 1,
                       bool auto_execute = false,
                       bool facing_required = true)
      : ActionConfig(required_resources, consumed_resources, priority, auto_execute),
        facing_required(facing_required) {}
};

class PutRecipeItems : public ActionHandler {
public:
  explicit PutRecipeItems(const PutItemsActionConfig& cfg)
      : ActionHandler(cfg, "put_items"), _facing_required(cfg.facing_required) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    Converter* converter = nullptr;

    if (_facing_required) {
      // Original behavior: must be facing the converter
      GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
      target_loc.layer = GridLayer::ObjectLayer;
      converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
    } else {
      // New behavior: can be next to the converter in any direction
      for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
          if (dr == 0 && dc == 0) continue;  // Skip the agent's own position
          GridLocation neighbor_loc = {actor->location.r + dr, actor->location.c + dc, GridLayer::ObjectLayer};
          Converter* neighbor_converter = dynamic_cast<Converter*>(_grid->object_at(neighbor_loc));
          if (neighbor_converter) {
            converter = neighbor_converter;
            break;
          }
        }
        if (converter) break;
      }
    }

    if (!converter) {
      return false;
    }

    bool success = false;
    for (const auto& [item, resources_required] : converter->input_resources) {
      if (actor->inventory.count(item) == 0) {
        continue;
      }
      InventoryQuantity resources_available = actor->inventory.at(item);
      InventoryQuantity resources_to_put = std::min(resources_required, resources_available);

      if (resources_to_put > 0) {
        InventoryDelta resources_put = converter->update_inventory(item, resources_to_put);
        if (resources_put > 0) {
          [[maybe_unused]] InventoryDelta delta = actor->update_inventory(item, -resources_put);
          assert(delta == -resources_put);
          actor->stats.add(actor->stats.resource_name(item) + ".put", resources_put);
          success = true;
        }
      }
    }

    return success;
  }

private:
  bool _facing_required;
};

namespace py = pybind11;

inline void bind_put_items_action_config(py::module& m) {
  py::class_<PutItemsActionConfig, ActionConfig, std::shared_ptr<PutItemsActionConfig>>(m, "PutItemsActionConfig")
      .def(py::init<const std::map<InventoryItem, InventoryQuantity>&,
                    const std::map<InventoryItem, InventoryQuantity>&,
                    unsigned char,
                    bool,
                    bool>(),
           py::arg("required_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("consumed_resources") = std::map<InventoryItem, InventoryQuantity>(),
           py::arg("priority") = 1,
           py::arg("auto_execute") = false,
           py::arg("facing_required") = true)
      .def_readwrite("facing_required", &PutItemsActionConfig::facing_required);
}

#endif  // ACTIONS_PUT_RECIPE_ITEMS_HPP_
