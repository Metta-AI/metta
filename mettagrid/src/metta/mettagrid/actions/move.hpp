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

    // Note: GridCoord is unsigned, so moving off the edge (e.g., 0 - 1) will underflow to a large value
    // (e.g., 65535 for uint16_t), and the move will fail at the is_valid_location check below since it will
    // be greater than the expected map dimensions. This assumes we will never have a map with width or height
    // equal to the max value of a GridCoord but we can save a few comparisons by making this assumption.
    target_location.r = static_cast<GridCoord>(static_cast<int>(target_location.r) + dr);
    target_location.c = static_cast<GridCoord>(static_cast<int>(target_location.c) + dc);

    // Update orientation to face the movement direction (even if movement fails)
    actor->orientation = move_direction;

    // Check if target location is valid and empty
    if (!_is_valid_square(target_location, _game_config->no_agent_interference)) {
      return false;
    }

    // Move the agent
    if (_game_config->no_agent_interference) {
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

private:
  const GameConfig* _game_config;
};

#endif  // ACTIONS_MOVE_HPP_
