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
    // Debug logging to see what parameter we're receiving
    printf("Move action received arg: %d\n", static_cast<int>(arg));
    
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

    // Calculate target location
    target_location.r += dr;
    target_location.c += dc;

    // Check if target location is valid and empty
    if (!_is_valid_square(target_location, _game_config->no_agent_interference)) {
      return false;
    }

    // Update orientation to face the movement direction
    actor->orientation = move_direction;

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
