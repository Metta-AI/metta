#ifndef ACTIONS_PLACE_BOX_HPP_
#define ACTIONS_PLACE_BOX_HPP_

#include <string>
#include <iostream>

#include "action_handler.hpp"
#include "grid.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/converter.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class PlaceBox : public ActionHandler {
public:
  std::map<InventoryItem, InventoryQuantity> resources_to_create;
  explicit PlaceBox(const ActionConfig& cfg, std::map<InventoryItem, InventoryQuantity> resources_to_create)
    : ActionHandler(cfg, "place_box"), resources_to_create(resources_to_create) {}

  unsigned char max_arg() const override {
    return 0;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg /*arg*/) override {
    GridLocation target_loc = _grid->relative_location(actor->location, static_cast<Orientation>(actor->orientation));
    target_loc.layer = GridLayer::ObjectLayer;

    if (!_grid->is_empty(target_loc.r, target_loc.c)) {
        return false;
    }

    bool can_create_box = true;
    for (const auto& [item, qty_required] : resources_to_create) {
        auto it = actor->inventory.find(item);
        if (it == actor->inventory.end()) {
            can_create_box = false;
            break;
        }
        if (it->second < qty_required) {
            can_create_box = false;
            break;
        }
    }
    if (!can_create_box) {
        return false;
    }

    for (const auto& [item, qty_required] : resources_to_create) {
        actor->update_inventory(item, -qty_required);
    }

    if (actor->box) {
        _grid->move_object(actor->box->id, target_loc);
    }

    actor->stats.add("box.created", 1.0f);
    return true;
  }
};

#endif  // ACTIONS_PLACE_BOX_HPP_
