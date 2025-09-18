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


class PutRecipeItems : public ActionHandler {
public:
  explicit PutRecipeItems(const ItemsActionConfig& cfg)
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
      // New behavior: can be next to the converter in cardinal directions only
      converter = _grid->next_to<Converter>(actor->location, GridLayer::ObjectLayer);
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
  // Alias for backwards compatibility - ItemsActionConfig is already bound in action_handler.hpp
  m.attr("PutItemsActionConfig") = m.attr("ItemsActionConfig");
}

#endif  // ACTIONS_PUT_RECIPE_ITEMS_HPP_
