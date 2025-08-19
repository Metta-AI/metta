#ifndef ACTIONS_MOVE_CARDINAL_HPP_
#define ACTIONS_MOVE_CARDINAL_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class MoveCardinal : public ActionHandler {
public:
  explicit MoveCardinal(const ActionConfig& cfg) : ActionHandler(cfg, "move_cardinal") {}

  unsigned char max_arg() const override {
    return 3;  // 4 cardinal directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Cardinal movement: direct movement in 4 cardinal directions
    Orientation move_direction;
    switch (arg) {
      case 0:  // North (Up)
        move_direction = Orientation::Up;
        break;
      case 1:  // South (Down)
        move_direction = Orientation::Down;
        break;
      case 2:  // West (Left)
        move_direction = Orientation::Left;
        break;
      case 3:  // East (Right)
        move_direction = Orientation::Right;
        break;
      default:
        return false;
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = _grid->relative_location(current_location, move_direction);

    // Update orientation to face movement direction
    actor->orientation = move_direction;

    // Determine if target has a box; move box out of the way if resources are satisfied
    GridLocation target_object_location = target_location;
    target_object_location.layer = GridLayer::ObjectLayer;
    GridObject* target_object = _grid->object_at(target_object_location);
    Box* target_box = dynamic_cast<Box*>(target_object);
    if (target_object) {
      if (!target_box) {
        return false;
      }
      bool has_resources = true;
      for (const auto& [item, qty] : target_box->resources_to_pick_up) {
        if (actor->inventory[item] < qty) { has_resources = false; break; }
      }
      if (!has_resources) {
        return false;
      }
      for (const auto& [item, qty] : target_box->resources_to_pick_up) {
        actor->update_inventory(item, -static_cast<InventoryDelta>(qty));
      }
      _grid->ghost_move_object(target_box->id, GridLocation(0, 0, GridLayer::ObjectLayer));
    }

    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }

    // Move the agent with new orientation
    return _grid->move_object(actor->id, target_location);
  }
};

#endif  // ACTIONS_MOVE_CARDINAL_HPP_
