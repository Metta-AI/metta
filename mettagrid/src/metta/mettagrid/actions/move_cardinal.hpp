#ifndef ACTIONS_MOVE_CARDINAL_HPP_
#define ACTIONS_MOVE_CARDINAL_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
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

    // Check if we are blocked by an obstacle
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }

    // Update orientation to face movement direction
    actor->orientation = move_direction;

    // Move the agent with new orientation
    return _grid->move_object(actor->id, target_location);
  }
};

#endif  // ACTIONS_MOVE_CARDINAL_HPP_
