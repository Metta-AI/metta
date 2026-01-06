#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "actions/action_handler.hpp"
#include "actions/attack.hpp"
#include "actions/move_config.hpp"
#include "actions/orientation.hpp"
#include "actions/transfer.hpp"
#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/usable.hpp"

struct GameConfig;

class Move : public ActionHandler {
public:
  explicit Move(const MoveActionConfig& cfg, [[maybe_unused]] const GameConfig* game_config)
      : ActionHandler(cfg, "move"), _allowed_directions(cfg.allowed_directions) {
    // Build direction name to orientation mapping
    _direction_map["north"] = Orientation::North;
    _direction_map["south"] = Orientation::South;
    _direction_map["west"] = Orientation::West;
    _direction_map["east"] = Orientation::East;
    _direction_map["northwest"] = Orientation::Northwest;
    _direction_map["northeast"] = Orientation::Northeast;
    _direction_map["southwest"] = Orientation::Southwest;
    _direction_map["southeast"] = Orientation::Southeast;
  }

  std::vector<Action> create_actions() override {
    std::vector<Action> actions;
    // Create actions in the order specified by the config
    for (const std::string& direction : _allowed_directions) {
      auto it = _direction_map.find(direction);
      if (it != _direction_map.end()) {
        actions.emplace_back(this, "move_" + direction, static_cast<ActionArg>(it->second));
      }
    }
    return actions;
  }

  void set_action_handlers(const std::unordered_map<std::string, ActionHandler*>& handlers) {
    _handlers = handlers;

    // Build vibe->handler map from handlers' vibes fields
    _vibe_handlers.clear();
    for (const auto& [name, handler] : handlers) {
      if (name == "attack") {
        Attack* attack = dynamic_cast<Attack*>(handler);
        if (attack) {
          for (ObservationType vibe : attack->get_vibes()) {
            _vibe_handlers[vibe] = handler;
          }
        }
      } else if (name == "transfer") {
        Transfer* transfer = dynamic_cast<Transfer*>(handler);
        if (transfer) {
          for (ObservationType vibe : transfer->get_vibes()) {
            _vibe_handlers[vibe] = handler;
          }
        }
      }
    }
  }

protected:
  bool _handle_action(Agent& actor, ActionArg arg) override {
    // Get the orientation from the action argument
    Orientation move_direction = static_cast<Orientation>(arg);

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

    if (!_grid->is_valid_location(target_location)) {
      return false;
    }

    // Get target object (may be nullptr if empty)
    GridObject* target_object = _grid->object_at(target_location);

    // Check if vibe-specific action override applies (from handler's vibes)
    auto vibe_handler_it = _vibe_handlers.find(actor.vibe);
    if (vibe_handler_it != _vibe_handlers.end()) {
      ActionHandler* handler = vibe_handler_it->second;
      // Let the handler decide if target is valid
      Attack* attack_handler = dynamic_cast<Attack*>(handler);
      if (attack_handler && attack_handler->try_attack(actor, target_object)) {
        return true;
      }
      Transfer* transfer_handler = dynamic_cast<Transfer*>(handler);
      if (transfer_handler && transfer_handler->try_transfer(actor, target_object)) {
        return true;
      }
    }

    // If location is empty, move
    if (_grid->is_empty(target_location.r, target_location.c)) {
      return _grid->move_object(actor, target_location);
    }

    // Swap with frozen agents (must check before usable since Agent is Usable)
    Agent* target_agent = dynamic_cast<Agent*>(target_object);
    if (target_agent && target_agent->frozen > 0) {
      bool swapped = _grid->swap_objects(actor, *target_agent);
      if (swapped) {
        actor.stats.incr("actions.swap");
      }
      return swapped;
    }

    // Try to use the object at target location
    if (target_object) {
      Usable* usable_object = dynamic_cast<Usable*>(target_object);
      if (usable_object) {
        return usable_object->onUse(actor, arg);
      }
    }

    return false;
  }

  std::string variant_name(ActionArg arg) const override {
    Orientation move_direction = static_cast<Orientation>(arg);
    return std::string(action_name()) + "_" + OrientationFullNames[static_cast<size_t>(move_direction)];
  }

private:
  std::vector<std::string> _allowed_directions;
  std::unordered_map<std::string, Orientation> _direction_map;
  std::unordered_map<std::string, ActionHandler*> _handlers;
  std::unordered_map<ObservationType, ActionHandler*> _vibe_handlers;  // vibe -> handler map
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_ACTIONS_MOVE_HPP_
