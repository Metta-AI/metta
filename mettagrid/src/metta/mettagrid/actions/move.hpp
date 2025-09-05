#ifndef ACTIONS_MOVE_HPP_
#define ACTIONS_MOVE_HPP_

#include <string>

#include "action_handler.hpp"
#include "grid_object.hpp"
#include "objects/agent.hpp"
#include "orientation.hpp"
#include "types.hpp"

// Forward declaration
struct GameConfig;

class Move : public ActionHandler {
public:
  explicit Move(const ActionConfig& cfg, const GameConfig* game_config)
      : ActionHandler(cfg, "move"), _game_config(game_config) {}

  unsigned char max_arg() const override {
    return _game_config->allow_diagonals ? 7 : 3;  // 8 directions if diagonals, 4 otherwise
  }

protected:
  bool _handle_action(Agent* actor, ActionArg arg) override {
    // Get the orientation from the action argument
    Orientation move_direction = static_cast<Orientation>(arg);

    // Validate the direction based on diagonal support
    if (!isValidOrientation(move_direction, _game_config->allow_diagonals)) {
      return false;
    }

    GridLocation current_location = actor->location;
    GridLocation target_location = current_location;

    // Get movement deltas for the direction
    int dc, dr;
    getOrientationDelta(move_direction, dc, dr);

    // Note: We currently expect all maps to have wall boundaries at the perimeter, so agents should not
    // be able to reach the edge coordinates (row/column 0 or max). If this changes someday and an agent
    // attempts to move off the edge of the map, the unsigned GridCoord would underflow to a large value
    // (e.g., 65535 for uint16_t). This underflow would likely be caught by the is_valid_location check
    // below, because we expect to never have a map with width or height equal to the max value of GridCoord.
    // We are not explicitly returning false for over/underflow because we want to avoid the extra comparisons
    // for performance.
    target_location.r = static_cast<GridCoord>(static_cast<int>(target_location.r) + dr);
    target_location.c = static_cast<GridCoord>(static_cast<int>(target_location.c) + dc);

    // Update orientation to face the movement direction (even if movement fails)
    actor->orientation = move_direction;

    // Check if target location is valid and empty
    if (!_is_valid_square(target_location)) {
      return false;
    }

    // Move the agent
    return _grid->move_object(actor->id, target_location);
  }

  bool _is_valid_square(GridLocation target_location) {
    if (!_grid->is_valid_location(target_location)) {
      return false;
    }
    if (!_grid->is_empty_at_layer(target_location.r, target_location.c, GridLayer::ObjectLayer)) {
      return false;
    }
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      return false;
    }
    return true;
  }

private:
  const GameConfig* _game_config;
};

#endif  // ACTIONS_MOVE_HPP_
