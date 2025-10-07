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
  Rotate(const ActionConfig& cfg, const GameConfig* game_config, Orientation orientation, const std::string& name)
      : ActionHandler(cfg, name), _game_config(game_config), _orientation(orientation) {}

protected:
  bool _handle_action(Agent& actor) override {
    if (!_game_config->allow_diagonals && isDiagonal(_orientation)) {
      return false;
    }

    actor.orientation = _orientation;

    if (_game_config->track_movement_metrics) {
      actor.stats.add(std::string("movement.rotation.to_") + OrientationNames[static_cast<int>(_orientation)], 1);
      if (actor.prev_action_name.rfind("rotate", 0) == 0) {
        actor.stats.add("movement.sequential_rotations", 1);
      }
    }

    return true;
  }

private:
  const GameConfig* _game_config;
  Orientation _orientation;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_ROTATE_HPP_
