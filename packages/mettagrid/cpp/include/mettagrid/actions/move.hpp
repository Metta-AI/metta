#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "actions/orientation.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/usable.hpp"

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
  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Get the orientation from the action argument
    Orientation move_direction = static_cast<Orientation>(arg);

    // Validate the direction based on diagonal support
    if (!isValidOrientation(move_direction, _game_config->allow_diagonals)) {
      return false;
    }

    GridLocation current_location = actor.location;
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
    actor.orientation = move_direction;

    if (!_grid->is_valid_location(target_location)) {
      return false;
    }

    // `Move` is actually `MoveOrUse`, so we need to check if the target location is empty and if the object is usable.
    // In the future, we may want to split 'Move' and 'MoveOrUse', if we want to allow agents to run into usable
    // objects without using them.
    if (!_grid->is_empty(target_location.r, target_location.c)) {
      // Check the AgentLayer first for other agents
      GridLocation agent_location = {target_location.r, target_location.c, GridLayer::AgentLayer};
      GridObject* target_agent = _grid->object_at(agent_location);
      if (target_agent) {
        Usable* usable_agent = dynamic_cast<Usable*>(target_agent);
        if (usable_agent) {
          return usable_agent->onUse(actor, arg);
        }
      }

      // Then check the ObjectLayer for other usable objects
      GridLocation object_location = {target_location.r, target_location.c, GridLayer::ObjectLayer};
      GridObject* target_object = _grid->object_at(object_location);
      if (target_object) {
        Usable* usable_object = dynamic_cast<Usable*>(target_object);
        if (usable_object) {
          return usable_object->onUse(actor, arg);
        }
      }

      return false;
    }

    // Move the agent
    return _grid->move_object(actor, target_location);
  }

  std::string variant_name(ActionArg arg) const override {
    Orientation move_direction = static_cast<Orientation>(arg);
    if (!isValidOrientation(move_direction, _game_config->allow_diagonals)) {
      return ActionHandler::variant_name(arg);
    }
    return std::string(action_name()) + "_" + OrientationFullNames[static_cast<size_t>(move_direction)];
  }

private:
  const GameConfig* _game_config;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_
