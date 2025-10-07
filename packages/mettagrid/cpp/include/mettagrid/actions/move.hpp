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
  Move(const ActionConfig& cfg, const GameConfig* game_config, Orientation orientation, const std::string& name)
      : ActionHandler(cfg, name), _game_config(game_config), _orientation(orientation) {}

protected:
  bool _handle_action(Agent& actor) override {
    if (!isValidOrientation(_orientation, _game_config->allow_diagonals)) {
      return false;
    }

    GridLocation current_location = actor.location;
    GridLocation target_location = current_location;

    int dc, dr;
    getOrientationDelta(_orientation, dc, dr);

    target_location.r = static_cast<GridCoord>(static_cast<int>(target_location.r) + dr);
    target_location.c = static_cast<GridCoord>(static_cast<int>(target_location.c) + dc);

    actor.orientation = _orientation;

    if (!grid().is_valid_location(target_location)) {
      return false;
    }

    if (!grid().is_empty(target_location.r, target_location.c)) {
      GridLocation agent_location = {target_location.r, target_location.c, GridLayer::AgentLayer};
      GridObject* target_agent = grid().object_at(agent_location);
      if (target_agent) {
        Usable* usable_agent = dynamic_cast<Usable*>(target_agent);
        if (usable_agent) {
          return usable_agent->onUse(actor);
        }
      }

      GridLocation object_location = {target_location.r, target_location.c, GridLayer::ObjectLayer};
      GridObject* target_object = grid().object_at(object_location);
      if (target_object) {
        Usable* usable_object = dynamic_cast<Usable*>(target_object);
        if (usable_object) {
          return usable_object->onUse(actor);
        }
      }

      return false;
    }

    return grid().move_object(actor, target_location);
  }

private:
  const GameConfig* _game_config;
  Orientation _orientation;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_
