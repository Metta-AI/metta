#ifndef ACTIONS_MOVE_8DIRECTIONAL_HPP_
#define ACTIONS_MOVE_8DIRECTIONAL_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Move8Directional : public ActionHandler {
public:
  explicit Move8Directional(const ActionConfig& cfg) : ActionHandler(cfg, "move_8directional") {}

  unsigned char max_arg() const override {
    return 7;  // 0-7 for 8 directions
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // 8-directional movement: agents move directly in any of 8 directions
    // WITHOUT changing orientation
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

    // Calculate target location based on direction
    switch (arg) {
      case 0:  // North
        target_location.r -= 1;
        break;
      case 1:  // Northeast
        target_location.r -= 1;
        target_location.c += 1;
        break;
      case 2:  // East
        target_location.c += 1;
        break;
      case 3:  // Southeast
        target_location.r += 1;
        target_location.c += 1;
        break;
      case 4:  // South
        target_location.r += 1;
        break;
      case 5:  // Southwest
        target_location.r += 1;
        target_location.c -= 1;
        break;
      case 6:  // West
        target_location.c -= 1;
        break;
      case 7:  // Northwest
        target_location.r -= 1;
        target_location.c -= 1;
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

    // Move the agent - orientation remains unchanged
    return _grid->move_object(actor->id, target_location);
  }
};

#endif  // ACTIONS_MOVE_8DIRECTIONAL_HPP_