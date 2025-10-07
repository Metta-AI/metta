#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ROTATE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ROTATE_HPP_

#include <string>

#include "actions/action_handler.hpp"
#include "actions/orientation.hpp"
#include "config/mettagrid_config.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"

class Rotate : public ActionHandler {
public:
  explicit Rotate(const ActionConfig& cfg, const GameConfig* game_config)
      : ActionHandler(cfg, "rotate"), _game_config(game_config) {}

  unsigned char max_arg() const override {
    return _game_config->allow_diagonals ? 7 : 3;  // 0-7 for 8 directions, 0-3 for 4 directions
  }

protected:
  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Validate the orientation argument
    Orientation orientation = static_cast<Orientation>(arg);
    if (!_game_config->allow_diagonals && isDiagonal(orientation)) {
      return false;
    }

    actor.orientation = orientation;

    // Track which orientation the agent rotated to (only if tracking enabled)
    if (_game_config->track_movement_metrics) {
      actor.stats.add(std::string("movement.rotation.to_") + OrientationNames[static_cast<int>(orientation)], 1);

      // Check if last action was also a rotation for sequential tracking
      if (actor.prev_action_name == _action_name) {
        actor.stats.add("movement.sequential_rotations", 1);
      }
    }

    return true;
  }

  std::string variant_name(ActionArg arg) const override {
    Orientation orientation = static_cast<Orientation>(arg);
    if (!_game_config->allow_diagonals && isDiagonal(orientation)) {
      return ActionHandler::variant_name(arg);
    }
    return std::string(action_name()) + "_" + OrientationFullNames[static_cast<size_t>(orientation)];
  }

private:
  const GameConfig* _game_config;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ROTATE_HPP_
