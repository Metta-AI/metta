#ifndef ACTIONS_MOVE_CARDINAL_HPP_
#define ACTIONS_MOVE_CARDINAL_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class MoveCardinal : public ActionHandler {
public:
  explicit MoveCardinal(const ActionConfig& cfg) : ActionHandler(cfg, "move") {}

  unsigned char max_arg() const override {
    return 3;  // 0 = North (Up), 1 = South (Down), 2 = West (Left), 3 = East (Right)
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Cardinal movement: agents move directly in cardinal directions
    // arg == 0: Move North (Up)
    // arg == 1: Move South (Down)
    // arg == 2: Move West (Left)
    // arg == 3: Move East (Right)

    // Map action argument to orientation/direction
    Orientation move_direction;
    switch (arg) {
      case 0:
        move_direction = Orientation::Up;
        break;
      case 1:
        move_direction = Orientation::Down;
        break;
      case 2:
        move_direction = Orientation::Left;
        break;
      case 3:
        move_direction = Orientation::Right;
        break;
      default:
        // Invalid argument, should be caught by max_arg() check
        return false;
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = _grid->relative_location(current_location, move_direction);

    // Check if we are blocked by an obstacle
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }

    return _grid->move_object(actor->id, target_location);
  }
};

#endif  // ACTIONS_MOVE_CARDINAL_HPP_