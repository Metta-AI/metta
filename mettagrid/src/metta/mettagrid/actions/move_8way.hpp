#ifndef ACTIONS_MOVE_8WAY_HPP_
#define ACTIONS_MOVE_8WAY_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "objects/box.hpp"
#include "types.hpp"

class Move8Way : public ActionHandler {
public:
  bool _no_agent_interference;
  explicit Move8Way(const ActionConfig& cfg, bool no_agent_interference = false)
  : ActionHandler(cfg, "move_8way"), _no_agent_interference(no_agent_interference) {}

  unsigned char max_arg() const override {
    return 7;  // 8 directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // 8-way movement: direct movement in 8 directions including diagonals
    GridLocation current_location = actor->location;
    GridLocation target_location = current_location;
    Orientation new_orientation = actor->orientation;

    switch (arg) {
      case 0:  // North
        target_location.r -= 1;
        new_orientation = Orientation::Up;
        break;
      case 1:  // Northeast
        target_location.r -= 1;
        target_location.c += 1;
        new_orientation = Orientation::Up;
        if (!_is_valid_square(target_location, _no_agent_interference)) {
            // Tries clockwise adj cardinal dir
            target_location.r += 1;
            new_orientation = Orientation::Up;
            if (!_is_valid_square(target_location, _no_agent_interference)) {
                // Tries counter-clockwise adj cardinal dir
                target_location.r -= 1;
                target_location.c -= 1;
                new_orientation = Orientation::Right;
                break;
            }
            break;
        }
        break;
      case 2:  // East
        target_location.c += 1;
        new_orientation = Orientation::Right;
        break;
      case 3:  // Southeast
        target_location.r += 1;
        target_location.c += 1;
        new_orientation = Orientation::Down;
        if (!_is_valid_square(target_location, _no_agent_interference)) {
            // Tries clockwise adj cardinal dir
            target_location.c -= 1;
            new_orientation = Orientation::Right;
            if (!_is_valid_square(target_location, _no_agent_interference)) {
                // Tries counter-clockwise adj cardinal dir
                target_location.r -= 1;
                target_location.c += 1;
                new_orientation = Orientation::Down;
                break;
            }
            break;
        }
        break;
      case 4:  // South
        target_location.r += 1;
        new_orientation = Orientation::Down;
        break;
      case 5:  // Southwest
        target_location.r += 1;
        target_location.c -= 1;
        new_orientation = Orientation::Down;
        if (!_is_valid_square(target_location, _no_agent_interference)) {
            // Tries clockwise adj cardinal dir
            target_location.r -= 1;
            new_orientation = Orientation::Down;
            if (!_is_valid_square(target_location, _no_agent_interference)) {
                // Tries counter-clockwise adj cardinal dir
                target_location.r += 1;
                target_location.c += 1;
                new_orientation = Orientation::Left;
                break;
            }
            break;
        }
        break;
      case 6:  // West
        target_location.c -= 1;
        new_orientation = Orientation::Left;
        break;
      case 7:  // Northwest
        target_location.r -= 1;
        target_location.c -= 1;
        new_orientation = Orientation::Up;
        if (!_is_valid_square(target_location, _no_agent_interference)) {
            // Tries clockwise adj cardinal dir
            target_location.c += 1;
            new_orientation = Orientation::Left;
            if (!_is_valid_square(target_location, _no_agent_interference)) {
                // Tries counter-clockwise adj cardinal dir
                target_location.r += 1;
                target_location.c -= 1;
                new_orientation = Orientation::Up;
                break;
            }
            break;
        }
        break;
      default:
        return false;
    }

    // Update orientation before moving
    actor->orientation = new_orientation;

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
        if (actor->inventory[item] < qty) {
          has_resources = false;
          break;
        }
      }
      if (!has_resources) {
        return false;
      }
      for (const auto& [item, qty] : target_box->resources_to_pick_up) {
        actor->update_inventory(item, -static_cast<InventoryDelta>(qty));
      }
      _grid->ghost_move_object(target_box->id, GridLocation(0, 0, GridLayer::ObjectLayer));
    }

    // Check if only/remaining target location is valid and empty
    if (!_is_valid_square(target_location, _no_agent_interference)) {
      return false;
    }

    // Move the agent with new orientation
    if (_no_agent_interference) {
      return _grid->ghost_move_object(actor->id, target_location);
    } else {
      return _grid->move_object(actor->id, target_location);
    }
  }
  bool _is_valid_square(GridLocation target_location, bool no_agent_interference) {
    if (!_grid->is_valid_location(target_location)) {
      return false;
    }
    if (no_agent_interference) {
      if (!_grid->is_empty_at_layer(target_location.r, target_location.c, GridLayer::ObjectLayer)) {
        return false;
      }
    } else {
      if (!_grid->is_empty(target_location.r, target_location.c)) {
        return false;
      }
    }
    return true;
  }
};

#endif  // ACTIONS_MOVE_8WAY_HPP_
