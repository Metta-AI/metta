#ifndef ACTIONS_GET_OUTPUT_HPP_
#define ACTIONS_GET_OUTPUT_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/box.hpp"
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
    Box* box = nullptr;

    if (_facing_required) {
      // Original behavior: must be facing the converter/box
      GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
      target_loc.layer = GridLayer::ObjectLayer;
      converter = dynamic_cast<Converter*>(_grid->object_at(target_loc));
      box = dynamic_cast<Box*>(_grid->object_at(target_loc));
    } else {
      // New behavior: can be next to the converter/box in cardinal directions only
      converter = _grid->next_to<Converter>(actor->location, GridLayer::ObjectLayer);
      box = _grid->next_to<Box>(actor->location, GridLayer::ObjectLayer);
    }

    if (!converter && !box) {
      return false;
    }
    // If converter, get output from converter
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

    // If box, pick up box if allowed
    if (box) {
      if (actor->agent_id == box->creator_agent_id) {
        // Creator cannot open their own box
        return false;
      }
      // If the creator of the box is an agent, return blue battery to creator and penalize creator
      if (box->creator_agent_id != 255) {
        Agent* creator = dynamic_cast<Agent*>(_grid->object(box->creator_agent_object_id));
        if (!creator) {
          return false;
        }
        // Return required resources to create box to creator inventory
        for (const auto& [item, amount] : box->returned_resources) {
          if (amount > 0) {
            creator->update_inventory(item, amount);
          }
        }
      }

      // Reward the agent for opening the box and teleport back to top-left corner
      _grid->ghost_move_object(box->id, GridLocation(0, 0, GridLayer::ObjectLayer));
      actor->stats.add("box.opened", 1.0f);
      return true;
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
