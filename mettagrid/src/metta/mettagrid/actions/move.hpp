#ifndef ACTIONS_MOVE_HPP_
#define ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Move : public ActionHandler {
public:
  bool _no_agent_interference;
  explicit Move(const ActionConfig& cfg, bool no_agent_interference = false)
      : ActionHandler(cfg, "move"), _no_agent_interference(no_agent_interference) {}

  unsigned char max_arg() const override {
    return 7;  // 8 directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // 8-way movement: direct movement in 8 directions including diagonals
    //
    // Movement direction mapping:
    // 7 0 1
    // 6 A 2
    // 5 4 3
    //
    // Final orientation: 0,1,7→Up  2,3→Right  4,5→Down  6→Left

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
        break;
      case 2:  // East
        target_location.c += 1;
        new_orientation = Orientation::Right;
        break;
      case 3:  // Southeast
        target_location.r += 1;
        target_location.c += 1;
        new_orientation = Orientation::Right;
        break;
      case 4:  // South
        target_location.r += 1;
        new_orientation = Orientation::Down;
        break;
      case 5:  // Southwest
        target_location.r += 1;
        target_location.c -= 1;
        new_orientation = Orientation::Down;
        break;
      case 6:  // West
        target_location.c -= 1;
        new_orientation = Orientation::Left;
        break;
      case 7:  // Northwest
        target_location.r -= 1;
        target_location.c -= 1;
        new_orientation = Orientation::Left;
        break;
      default:
        return false;
    }

    // Check if target location is valid and empty
    if (!_is_valid_square(target_location, _no_agent_interference)) {
      return false;
    }

    // Update orientation before moving
    actor->orientation = new_orientation;

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

#endif  // ACTIONS_MOVE_HPP_
