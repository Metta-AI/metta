#ifndef ACTIONS_GET_OUTPUT_HPP_
#define ACTIONS_GET_OUTPUT_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "types.hpp"


class GetOutput : public ActionHandler {
public:
  explicit GetOutput(const ItemsActionConfig& cfg)
      : ActionHandler(cfg, "get_items"), _facing_required(cfg.facing_required) {}

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
    // Get output from converter
    if (converter) {
      if (!converter->inventory_is_accessible()) {
        return false;
      }

      // Actions is only successful if we take at least one item.
      bool resources_taken = false;

      for (const auto& [item, _] : converter->output_resources) {
        if (converter->inventory.count(item) == 0) {
          continue;
        }
        InventoryDelta resources_available = converter->inventory[item];

        InventoryDelta taken = actor->update_inventory(item, resources_available);

        if (taken > 0) {
          actor->stats.add(actor->stats.resource_name(item) + ".get", taken);
          converter->update_inventory(item, -taken);
          resources_taken = true;
        }
      }
      return resources_taken;
    }
    return false;
  }

private:
  bool _facing_required;
};

namespace py = pybind11;

inline void bind_get_items_action_config(py::module& m) {
  // Alias for backwards compatibility - ItemsActionConfig is already bound in action_handler.hpp
  m.attr("GetItemsActionConfig") = m.attr("ItemsActionConfig");
}

#endif  // ACTIONS_GET_OUTPUT_HPP_
