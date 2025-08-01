#ifndef ACTIONS_MOVE_8WAY_HPP_
#define ACTIONS_MOVE_8WAY_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Move8Way : public ActionHandler {
public:
  explicit Move8Way(const ActionConfig& cfg) : ActionHandler(cfg, "move_8way") {}

  unsigned char max_arg() const override {
    return 7;  // 0-7 for 8 directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // 8-way movement: agents move directly in any of 8 directions
    // and update their orientation to face the direction of movement
    // arg == 0: North (Up)
    // arg == 1: Northeast
    // arg == 2: East (Right)
    // arg == 3: Southeast
    // arg == 4: South (Down)
    // arg == 5: Southwest
    // arg == 6: West (Left)
    // arg == 7: Northwest

    GridLocation current_location = actor->location;
    GridLocation target_location = current_location;
    Orientation new_orientation = actor->orientation;

    // Calculate target location and new orientation based on direction
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
        new_orientation = Orientation::Down;
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
        new_orientation = Orientation::Up;
        break;
      default:
        // Invalid argument
        return false;
    }

    // Check if target location is valid and empty
    if (!_grid->is_valid_location(target_location)) {
      return false;
    }

    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }

    // Update orientation before moving
    actor->orientation = new_orientation;

    // Move the agent with new orientation
    return _grid->move_object(actor->id, target_location);
  }
};

#endif  // ACTIONS_MOVE_8WAY_HPP_
