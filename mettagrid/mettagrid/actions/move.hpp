#ifndef METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_
#define METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/freeze_tile.hpp"

class Move : public ActionHandler {
public:
  explicit Move(const ActionConfig& cfg) : ActionHandler(cfg, "move") {}

  unsigned char max_arg() const override {
    return 1;
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    unsigned short direction = arg;

    Orientation target_orientation = static_cast<Orientation>(actor->orientation);

    // If agent is on a freeze tile, force movement in the stored direction
    if (actor->on_freeze_tile) {
      target_orientation = static_cast<Orientation>(actor->freeze_tile_direction);
    } else {
      // Normal movement - apply direction argument
      if (direction == 1) {
        if (target_orientation == Orientation::Up) {
          target_orientation = Orientation::Down;
        } else if (target_orientation == Orientation::Down) {
          target_orientation = Orientation::Up;
        } else if (target_orientation == Orientation::Left) {
          target_orientation = Orientation::Right;
        } else if (target_orientation == Orientation::Right) {
          target_orientation = Orientation::Left;
        }
      }
    }

    GridLocation old_loc = actor->location;
    GridLocation new_loc = _grid->relative_location(old_loc, target_orientation);

    // Check if target location is blocked
    if (!_grid->is_empty(new_loc.r, new_loc.c)) {
      // If agent was on a freeze tile and can't move, release from freeze tile
      if (actor->on_freeze_tile) {
        actor->on_freeze_tile = false;
        actor->freeze_tile_direction = 0;
      }
      return false;
    }

    // Perform the movement
    bool movement_successful = _grid->move_object(actor->id, new_loc);

    if (movement_successful) {
      // Check if the new location has a freeze tile
      GridLocation freeze_tile_loc = new_loc;
      freeze_tile_loc.layer = GridLayer::Object_Layer;
      GridObject* obj_at_new_loc = _grid->object_at(freeze_tile_loc);

      if (obj_at_new_loc && obj_at_new_loc->_type_id == ObjectType::FreezeTileT) {
        // Agent stepped on a freeze tile - store the direction they were moving
        actor->on_freeze_tile = true;
        actor->freeze_tile_direction = static_cast<unsigned char>(target_orientation);
      } else {
        // Agent moved to a location without a freeze tile
        actor->on_freeze_tile = false;
        actor->freeze_tile_direction = 0;
      }
    }

    return movement_successful;
  }
};

#endif  // METTAGRID_METTAGRID_ACTIONS_MOVE_HPP_
