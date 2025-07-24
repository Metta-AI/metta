#ifndef ACTIONS_MOVE_HPP_
#define ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "types.hpp"

class Move : public ActionHandler {
public:
  explicit Move(const ActionConfig& cfg, bool track_movement_metrics = false)
      : ActionHandler(cfg, "move"), _track_movement_metrics(track_movement_metrics) {}

  unsigned char max_arg() const override {
    return 1;  // 0 = move forward, 1 = move backward
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Move action: agents move in a direction without changing orientation
    // arg == 0: Move forward in current direction
    // arg == 1: Move backward (reverse direction) while maintaining facing direction

    Orientation move_direction = static_cast<Orientation>(actor->orientation);

    if (arg == 1) {
      move_direction = get_opposite_direction(move_direction);
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = _grid->relative_location(current_location, move_direction);

    // check if we are blocked by an obstacle
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }

    bool success = _grid->move_object(actor->id, target_location);

    // Track movement direction on success (only if tracking enabled)
    if (success && _track_movement_metrics) {
      static const char* direction_names[] = {"up", "down", "left", "right"};
      actor->stats.add(std::string("movement.direction.") + direction_names[static_cast<int>(move_direction)], 1);
    }

    return success;
  }

private:
  bool _track_movement_metrics;

  // Get the opposite direction (for backward movement)
  static Orientation get_opposite_direction(Orientation orientation) {
    switch (orientation) {
      case Orientation::Up:
        return Orientation::Down;
      case Orientation::Down:
        return Orientation::Up;
      case Orientation::Left:
        return Orientation::Right;
      case Orientation::Right:
        return Orientation::Left;
      default:
        assert(false && "Invalid orientation passed to get_opposite_direction()");
    }
  }
};

#endif  // ACTIONS_MOVE_HPP_
